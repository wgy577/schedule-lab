"""R13 JOINT-AGENTIC M2+M3 ROLLING GRPO (T1-M2-M3-JOINT-AGENTIC-ROLLING-GRPO-R13).

PERMANENT structure fixed by spec §0/§49 (identical for training AND evaluation):
    S_t + Appearance -> M2 root-candidate policy -> real root probe ->
    makespan-first root filter -> Memory rescue (only when no gain) ->
    Reasoner (frozen Contributor+Enabler; dependency-completed) ->
    complete legal Proposal pool -> M3 -> execute -> S_{t+1}

Responsibilities (§0): M2's ONLY job is to find the roots worth expanding next
(budgeted search); the Reasoner expands those roots into a complete legal Proposal
pool; M3 selects one Proposal (or STOP). M2 never chooses the final Proposal.

MAKESPAN FIRST (§1-2) -- the filter uses real FixedDecisionReplay gain ONLY:
    G_direct(u) = max real gain over single proposals rooted at op u
    G_dep(u)    = max real gain over dependency-completed PAIR proposals containing u
                  (contributor + enabler; §6/§48: X:M8->M5 direct<=0 must still try
                  the minimal dependence completion Y:M5->M6 + X:M8->M5 pair probe)
    G_probe(u)  = max(valid G_direct, valid G_dep)
    Tier A  PROVEN_GAIN     G_probe > 0     (never capped; Memory cannot veto, §2/§5)
    Tier B  MEMORY_RESCUED  G_probe <= 0 and Memory supports  (cap B_MEMORY, §17)
    Tier C  UNSUPPORTED     otherwise       -> pruned from the Proposal pool
Memory is second-level evidence ONLY -- NOT reward / NOT causal truth / NOT posterior /
NOT legality / NOT oracle / NOT final action authority (§5).

M2 root policy (§9-15): root_score_RL(u) = root_score_M2(u) + alpha_M2*tanh(delta_theta(u)),
root_score_M2(u) = frozen B5 attribution op_b5[u], alpha_M2 = 0.2 (small, zero-init MLP so
delta==0 IS the frozen attribution). Budget B_ROOT = 8 = 3 anchors (top attribution,
always, safety) + 4 policy sequential without-replacement draws (EXACT conditional
logprob, stored per draw with its remaining-set ids) + 1 uniform outsider. Only the
policy-drawn roots carry an M2 GRPO ratio (§15); probing/filtering are deterministic
and never produce gradient or reward (§16,24).

Joint GRPO (§25-29): one shared group advantage A_i = (R_i-mean)/(std+eps) serves both
the M2 root draws and the M3 actions of the same trajectory.
    L = lambda_M2*L_M2 + lambda_M3*L_M3 + beta_M2*KL(M2||attr-prior) + beta_M3*KL(M3||R6)
lambda_M2=0.5, lambda_M3=1.0, beta_M2=0.3 > beta_M3=0.03 (fixed).

Stage curriculum (§32-36): A = freeze M3, train adapter only; B = freeze adapter,
M3 rolling GRPO on the gated pool; C = JOINT (small M2 residual + strong KL, M3 normal).

identified=false, formal_test_access=0. Parent M3 = m3_rolling_grpo_v1.pt (R11 A),
alpha_M3 = 0.5 kept (§19-20).
"""
from __future__ import annotations

import copy
import json
import math
import os
import pickle
import random
import time
from collections import Counter, deque

import numpy as np
import torch
from torch import nn

from causal_schedule_lab.intervention import ScheduleGraphView
from causal_schedule_lab.validation import schedule_hash

from . import config as C
from .config import (
    TO1_R13_ALPHA_M2,
    TO1_R13_ALPHA_PROP,
    TO1_R13_ALPHA_STOP,
    TO1_R13_ANCHOR_ROOTS,
    TO1_R13_BETA_M2,
    TO1_R13_BETA_M3,
    MEM_FEAT_DIM,
    TO1_R13_CLIP_EPS,
    TO1_R13_HORIZON,
    TO1_R13_K,
    TO1_R13_LAMBDA_M2,
    TO1_R13_LAMBDA_M3,
    TO1_R13_LR_M2,
    TO1_R13_LR_M3,
    TO1_R13_M2_FEAT_DIM,
    TO1_R13_MAX_DEPTH,
    TO1_R13_MEMORY_BUDGET,
    TO1_R13_MEM_SUCCESS_MIN,
    TO1_R13_MEM_SUPPORT_MIN,
    TO1_R13_MIX_EPS,
    TO1_R13_OUTSIDER,
    TO1_R13_POLICY_DRAWS,
    TO1_R13_PROBE_PER_ROOT,
    TO1_R13_ROOT_BUDGET,
    TO1_R13_TEMP,
    TO1_R13_TEMP_M2,
    TO1_R13_UPDATE_EPOCHS,
)
from .sibling_diversity import (
    BoundedReplayCache, canonical_root_set, policy_context_fingerprint,
    rotated_action_sample, sibling_diversity,
)
from . import upstream
from .proposal_features import (
    AnalyzeCache,
    _edits_for,
    _execute_step,
    _execute_step_with_reason,
    _compact_root_appearance_context,
    _touched_operations,
    selected_root_paths,
    proposal_identity,
    old_base_of,
    state_feature_vec,
)
from .rolling_grpo import (
    M3RollingGRPOPolicy,
    _kl_mixture,
    _mem_record_count,
    mixture_logp,
    mixture_pmf,
    mixture_sample,
)
from .ranking import _rerank_feats_all, wide_pool
from .scorer import _scores
from .memory import ProgressiveMemory
from .top1 import _pool_stats_from, _rollex_of
from .traj_grpo import (
    _group_identity_key,
    _resolve_mp_ctx,
    _traj_seed,
    group_advantages_r12,
    selector_action_logits,
    unified_parity_closed_loop,
    unified_parity_rollout,
)
from .upstream import load_upstream


def _worker_heartbeat(stage, *, iid=None, traj_id=None, step=None, **details):
    """Publish a tiny per-process progress record for live stall diagnosis.

    The directory is unique to one training run.  Atomic replacement means the
    parent can safely read these files while workers update them.  Heartbeats do
    not enter trajectories, rewards, seeds, or model inputs.
    """
    directory = os.environ.get("T2M_WORKER_HEARTBEAT_DIR", "")
    if not directory or os.environ.get("T2M_WORKER_HEARTBEAT", "1") == "0":
        return
    try:
        os.makedirs(directory, exist_ok=True)
        pid = os.getpid()
        row = {
            "pid": pid,
            "wall_time": time.time(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": str(stage),
            "instance_id": None if iid is None else str(iid),
            "trajectory_id": None if traj_id is None else int(traj_id),
            "step": None if step is None else int(step),
            **details,
        }
        target = os.path.join(directory, f"worker_{pid}.json")
        temporary = target + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(row, stream, sort_keys=True)
        os.replace(temporary, target)
    except Exception:
        # Diagnostics must never change the training path.
        return


def _worker_heartbeat_summary():
    """Return the newest live stage for each worker, grouped for one log line."""
    directory = os.environ.get("T2M_WORKER_HEARTBEAT_DIR", "")
    if not directory or not os.path.isdir(directory):
        return "unavailable"
    now = time.time()
    rows = []
    for name in os.listdir(directory):
        if not name.startswith("worker_") or not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as stream:
                row = json.load(stream)
            pid = int(row.get("pid", -1))
            if pid <= 0:
                continue
            try:
                os.kill(pid, 0)
            except OSError:
                continue
            age = max(0.0, now - float(row.get("wall_time", now)))
            rows.append((age, row))
        except Exception:
            continue
    if not rows:
        return "empty"
    rows.sort(reverse=True, key=lambda item: item[0])
    return ";".join(
        f"pid={row.get('pid')}:{row.get('stage')}:{row.get('instance_id')}:"
        f"traj={row.get('trajectory_id')}:step={row.get('step')}:age={age:.0f}s"
        for age, row in rows[:24]
    )


def _force_shutdown_process_pool(executor):
    """Terminate a stalled ProcessPoolExecutor without waiting forever."""
    processes = list((getattr(executor, "_processes", None) or {}).values())
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except Exception:
            pass
    deadline = time.monotonic() + 5.0
    for process in processes:
        try:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        except Exception:
            pass
    for process in processes:
        try:
            if process.is_alive():
                process.kill()
        except Exception:
            pass
from .hierarchical_residual import trajectory_context

_T13_MP_CTX = "spawn"


def get_device(pref=None):
    """Auto-detect compute device (T2-B portability anchor).

    pref: 'auto'|'cpu'|'cuda' (defaults to config TO1_T2B_DEVICE).
    'auto' -> cuda iff available, else cpu (Mac dev machine has no CUDA).
    T2-B v1 does NOT move models to device (they are ~300 params and the loop is
    CPU-bound); this helper exists so a future CUDA-forward path has a single
    place to resolve the device.  A caller that does `.to(get_device())` should
    wrap the call in try/except torch.cuda.OutOfMemoryError -> cpu fallback.
    """
    p = (C.TO1_T2B_DEVICE if pref is None else pref)
    if p == "cuda":
        return torch.device("cuda")
    if p == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# M2RootPolicyAdapter -- zero-init residual over frozen B5 attribution  (§9,11)
# ---------------------------------------------------------------------------
class M2RootPolicyAdapter(nn.Module):
    """delta_root(u;theta) = MLP(root_features); small two-layer ReLU-style net with a
    ZERO-INIT final linear so delta==0 at warm start -> root_score_RL == frozen op_b5.

    root_score_RL(u) = root_base(u) + alpha_M2 * tanh(delta_theta(u))   (alpha_M2=0.2)
    root_base(u)     = row-standardized frozen attribution (T_M2-scale invariant prior).
    Only the adapter's params are trainable under joint GRPO -- the B5 attribution
    backend (encoder / per_block_candidate_score / causal-graph logic) stays frozen.
    """

    def __init__(self, feat_dim=8, alpha_m2=0.2):
        super().__init__()
        self.alpha_m2 = float(alpha_m2)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 16), nn.Tanh(), nn.Linear(16, 1))
        self._zero()

    def _zero(self):
        """Zero the final linear (weight+bias): net output is 0 -> delta==0."""
        with torch.no_grad():
            final = self.net[-1]
            final.weight.zero_()
            final.bias.zero_()

    def delta(self, F):
        """F [N, feat] -> residual pre-tanh [N,1]."""
        return self.net(F)

    def residual(self, F):
        """alpha_M2 * tanh(delta)  [N]."""
        return float(self.alpha_m2) * torch.tanh(self.net(F).squeeze(-1))

    def score(self, base, F):
        """root score [N] = standardized base + alpha*tanh(delta)."""
        return base + self.residual(F)

    def snapshot(self):
        with torch.no_grad():
            return {n: p.detach().clone() for n, p in self.named_parameters()}

    def load_snapshot(self, snap):
        with torch.no_grad():
            for n, p in self.named_parameters():
                if n in snap:
                    p.copy_(snap[n])


# ---------------------------------------------------------------------------
# root candidates / features -- frozen inputs to the adapter  (§12)
# ---------------------------------------------------------------------------
def _root_candidate_arrays(ast):
    """Deterministic candidate list + features + standardized base, cached per state.

    Candidate set = M2-attributed ops (op_b5, desc) THEN enabler-only ops (ops that
    appear as ENABLER edits in the pool but have no own attribution). Every candidate
    is probeable; the frozen attribution forms the policy's prior.

    Features (TO1_R13_M2_FEAT_DIM = 8), all normalized:
      [0] attribution        op_b5[u] / max_b5
      [1] block spread       op_n_blocks[u] / max_blocks
      [2] is_enabler         (no own attribution)
      [3] start_norm         interval.start / makespan   (earliest board first)
      [4] spent_norm         (end-start) / makespan
      [5] machine_util       machine busy time / makespan
      [6] in_contrib_pool    appears as a CONTRIBUTOR edit in the pool
      [7] prior_rank_norm    rank in desc op_b5 ordering / n_cand
    base(raw)   = op_b5[u] (0 for enabler-only) -> standardized across ALL candidates;
                  enabler-only rows get a floor one sigma below the weakest contributor
                  so the prior never puts an un-attributed op above an attested one.
    Returns dict with op list, feature matrix F [N,8], base [N], is_enabler mask.
    """
    hit = ast.get("_r13_cands")
    if hit is not None:
        return hit
    gv = ScheduleGraphView.from_problem_schedule(ast["problem"], ast["schedule"])
    machine_busy = Counter()
    for iv in gv.intervals:
        machine_busy[iv.machine_id] += iv.end - iv.start
    op_b5, op_n_blocks = ast["op_b5"], ast["op_n_blocks"]
    makespan = float(ast["schedule"].makespan)
    max_b5 = max(op_b5.values(), default=1.0)
    max_blk = max(op_n_blocks.values(), default=1)
    contrib_order = sorted(op_b5, key=lambda o: (-op_b5[o], o))
    rank_map = {o: i for i, o in enumerate(contrib_order)}
    contrib_set = set(contrib_order)
    enab_set = {rec["e"].operation_id for rec in ast["pool"]} - contrib_set
    enab_order = sorted(enab_set)
    ops = contrib_order + enab_order
    n_cand = len(ops)
    feats = np.zeros((n_cand, TO1_R13_M2_FEAT_DIM), dtype=np.float32)
    base_raw = np.zeros(n_cand, dtype=np.float64)
    enab_mask = torch.zeros(n_cand, dtype=torch.bool)
    for i, op in enumerate(ops):
        iv = gv.interval_for(op)
        start = iv.start if iv else 0.0
        dur = (iv.end - iv.start) if iv else 0.0
        mach = iv.machine_id if iv else ""
        util = machine_busy.get(mach, 0.0) / max(makespan, 1.0)
        att = op_b5.get(op, 0.0)
        blk = op_n_blocks.get(op, 0)
        feats[i] = [
            att / max_b5, min(blk / max_blk, 1.0),
            float(op in enab_set), start / max(makespan, 1.0),
            min(dur / max(makespan, 1.0), 1.0), min(util, 1.0),
            float(op in contrib_set),
            rank_map.get(op, len(contrib_order)) / max(n_cand, 1),
        ]
        base_raw[i] = att
        enab_mask[i] = bool(op in enab_set)
    base_std = base_raw.copy()
    # A rolled state may legitimately have no contributor/enabler roots.  Keep
    # the canonical empty tensors and let the gate return an empty pool/STOP;
    # reducing an empty NumPy array only emits warnings and cannot add signal.
    if n_cand:
        mu, sd = base_std.mean(), base_std.std()
        base_std = (base_std - mu) / (sd + 1e-6)
    # T2-I: no artificial "weakest contributor - 1 sigma" floor.  Zero B5
    # evidence remains zero before standardization and `is_enabler` stays an
    # explicit feature; RL, not a hand rule, decides whether it is useful.
    # T2-D: every root token reuses the frozen B5 Stream-C operation latent.
    # This is a decision-time tensor already computed by AnalyzeCache; absence is
    # an alignment bug and must not be hidden behind a fabricated embedding.
    latents = []
    for op in ops:
        ni = ast["node_index"].get(f"operation:{op}")
        if ni is None:
            raise RuntimeError(f"B5 operation latent missing for root {op!r}")
        latents.append(ast["h_c"][ni].detach().float())
    latent_dim = int(ast["h_c"].shape[-1]) if ast["h_c"].dim() == 2 else 0
    latent_t = (torch.stack(latents) if latents else
                torch.zeros((0, latent_dim), dtype=torch.float32))
    from .probability_trace import build_trace_graph
    trace_graph = build_trace_graph(ast, ops)
    app = _compact_root_appearance_context(ast, ops)
    out = {"ops": ops, "feats": torch.tensor(feats), "base": torch.tensor(
        base_std, dtype=torch.float32), "enab_mask": enab_mask, "n_cand": n_cand,
        "latents": latent_t.detach(), "latent_dim": latent_dim,
        "trace_graph": trace_graph,
        "app_raw": app["raw"], "app_mask": app["mask"],
        "app_raw_dim": int(app["raw_dim"]), "min_hop": app["min_hop"],
        "support_count": app["support_count"],
        "support_blocks": app["support_blocks"]}
    # The compact [root, <=8, relation] representation is the only form that
    # must survive candidate construction.  Drop the full [appearance,node,H]
    # tensors from the per-state cache so long rollouts do not retain them.
    for key in ("appearance_h_a", "appearance_q", "per_block_candidate_scores",
                "root_appearance_hops", "appearance_features"):
        ast.pop(key, None)
    ast["_r13_cands"] = out
    return out


def _sequential_draws(adapter, cand, pool_mask, n_draws, rng):
    """Exact sequential without-replacement sampling over the policy pool.

    Each draw stores (full-space index, remaining-id list BEFORE the draw, exact
    conditional logp). The adapter is FROZEN at collect time (behavior); the stored
    logp is the pi_old baseline the joint GRPO ratio is taken against (§14-15).
    """
    cand_f, cand_base = cand["feats"], cand["base"]
    T = float(C.TO1_R13_TEMP_M2)
    rem = list(pool_mask)               # pool_mask is an index list (§14 policy pool)
    draws = []
    for _ in range(n_draws):
        if not rem:
            break
        ridx = torch.tensor(rem, dtype=torch.long)
        with torch.no_grad():
            if hasattr(adapter, "final_logits_from_candidates"):
                raw_logits = adapter.final_logits_from_candidates(cand, ridx)
            else:
                delta = adapter.net(cand_f[ridx])[:, 0]
                raw_logits = cand_base[ridx] + float(TO1_R13_ALPHA_M2) * torch.tanh(delta)
        if '_explore_bias' in cand:
            raw_logits=raw_logits+cand['_explore_bias'][ridx].to(raw_logits.device)
        logits = raw_logits / T
        p_t = torch.softmax(logits, dim=0)
        eps2 = float(C.TO1_R13_MIX_EPS)
        p_t = (1.0 - eps2) * p_t + eps2 / max(len(rem), 1)
        p = p_t.detach().numpy()
        sel = int(rng.choices(range(len(rem)), weights=p, k=1)[0])
        entropy = float((-(p_t * torch.log(p_t.clamp_min(1e-12))).sum()).item())
        draws.append({"idx": rem[sel], "rem_ids": tuple(rem),
                      "logp": float(np.log(p[sel])),
                      "policy_entropy": entropy,
                      "effective_roots": float(np.exp(entropy)),
                      "m2_mix_eps": eps2})
        rem.pop(sel)
    return draws


# ---------------------------------------------------------------------------
# the M2 gate: budget -> REAL probe -> makespan-first tiers -> Proposal pool
# ---------------------------------------------------------------------------
def _probe_op(ast, metas, prop_feats, op, executor, base_ms, base_hash, cap):
    """Real FDR probes for root op u. Returns (G_direct, G_dep, n_exec, n_pos).

    singles: proposal metas whose edit.operation_id == u  -> G_direct(u)
    pairs:   composite (dependency-completed) metas containing u -> G_dep(u), credited
             ONLY to contributor ops (u in op_b5) -- enabler-only ops keep G_dep = 0.
    Proposal order within each group follows the existing learned prior
    (uhat idx296 for singles / direct idx298 for pairs) so the most promising edits
    execute first. Cap = TO1_R13_PROBE_PER_ROOT executions per group.
    """
    pool = ast["pool"]
    singles, pairs = [], []
    for k, m in enumerate(metas):
        if m["kind"] == "single":
            if pool[m["i"]]["e"].operation_id == op:
                singles.append((k, m, float(prop_feats[k, 296])))
        else:
            for _ix in (m["i"], m["j"]):
                if pool[_ix]["e"].operation_id == op:
                    pairs.append((k, m, float(prop_feats[k, 298])))
                    break
    singles.sort(key=lambda z: -z[2])
    pairs.sort(key=lambda z: -z[2])
    G1 = 0.0
    n_s = n_p = n_pos = 0
    for k, m, _ in singles[:cap]:
        res = _execute_step(executor, ast["problem"], ast["schedule"],
                            [pool[m["i"]]["e"]], base_ms, base_hash)
        n_s += 1
        if res is not None:
            g = float(res["improvement"])
            if g > 0:
                n_pos += 1
            G1 = max(G1, g)
    is_contrib = op in ast["op_b5"]
    Gd = 0.0
    for k, m, _ in pairs[:cap]:
        if not is_contrib:
            break
        edits = [pool[m["i"]]["e"], pool[m["j"]]["e"]]
        res = _execute_step(executor, ast["problem"], ast["schedule"], edits, base_ms, base_hash)
        n_p += 1
        if res is not None:
            g = float(res["improvement"])
            if g > 0:
                n_pos += 1
            Gd = max(Gd, g)
    return G1, Gd, n_s + n_p, n_pos


def _tier_label(gprobe, mem_hit):
    if gprobe > 0:
        return "A"
    return "B" if mem_hit else "C"


def _mem_evidence(pm, iid, episode_id, step, sf, ast, metas, prop_feats, op, cap):
    """Memory support for op u under a no-gain probe: query the best-prior single
    proposal's (type, role, src, tgt) fine key. Returns (support, success, mean_gain)."""
    pool = ast["pool"]
    best = None
    for k, m in enumerate(metas):
        if m["kind"] == "single" and pool[m["i"]]["e"].operation_id == op:
            row = float(prop_feats[k, 296])
            if best is None or row > best[0]:
                best = (row, m, k)
    if best is None:
        return 0.0, 0.0, 0.0
    m = best[1]
    q = {"type": "single", "role": pool[m["i"]]["role"],
         "src": pool[m["i"]]["e"].source_machine,
         "tgt": pool[m["i"]]["e"].target_machine}
    feats = pm.features(iid, episode_id, step, sf, [q])[0]
    return float(feats[0]), float(feats[1]), float(feats[2])


def _root_memory_features(pm, iid, episode_id, step, sf, ast, metas,
                          prop_feats, ops):
    """Decision-time Memory evidence only; never legality, truth, or reward."""
    root_ms = max(float(ast["schedule"].makespan), 1.0)
    rows, raw = [], {}
    for op in ops:
        support, success, mean_gain = _mem_evidence(
            pm, iid, episode_id, step, sf, ast, metas, prop_feats, op,
            int(C.TO1_R13_PROBE_PER_ROOT))
        raw[op] = (support, success, mean_gain)
        confidence = support / (support + float(C.TO1_R14_MEM_CONF_SCALE))
        rows.append([confidence, min(max(success, 0.0), 1.0),
                     math.tanh(mean_gain / root_ms)])
    return torch.tensor(rows, dtype=torch.float32), raw


def _m2_gate_step(ast, metas, prop_feats, adapter, executor, pm, iid, episode_id,
                  step, sf, rng):
    """The §0 permanent pipeline between Appearance and M3: budget -> probe -> tiers.

    Returns dict:
      gated_metas / gated_prop_feats   (subset kept, in original order)
      m2_rec  (per-step training record: cand arrays + anchors + draws + outsider)
      diag    (per-state counts for §42 diagnostics + Stage-A PASS gates)
      retained (sorted op ids)
    Also executes real probe replays (deterministic; no gradient, no reward).
    """
    cand = dict(_root_candidate_arrays(ast))
    cand["state_context"] = torch.as_tensor(sf, dtype=torch.float32).detach()
    n_cand = cand["n_cand"]
    ops = cand["ops"]
    cand["memory_evidence"], memory_raw = _root_memory_features(
        pm, iid, episode_id, step, sf, ast, metas, prop_feats, ops)
    cand["memory_evidence"] = cand["memory_evidence"].detach()
    anchors = []
    for i in range(n_cand):
        if not bool(cand["enab_mask"][i]):
            anchors.append(i)
    anchors = anchors[:int(C.TO1_R13_ANCHOR_ROOTS)]
    anchor_set = set(anchors)
    pool_mask = [i for i in range(n_cand) if i not in anchor_set]
    draws = _sequential_draws(adapter, cand, pool_mask,
                              int(C.TO1_R13_POLICY_DRAWS), rng)
    drawn = {d["idx"] for d in draws}
    outsider_pool = [i for i in pool_mask if i not in drawn]
    outsider = (int(rng.choice(outsider_pool)) if outsider_pool else -1)

    probe = {}
    tiers = {}
    pos_probe_cnt = 0
    best_probe_gain = 0.0
    mem_rescued = 0
    order = [i for i in anchors] + [d["idx"] for d in draws]
    if outsider >= 0:
        order.append(outsider)
    base_ms = int(ast["schedule"].makespan)
    base_hash = schedule_hash(ast["schedule"])
    for i in order:
        op = ops[i]
        G1, Gd, nex, npos = _probe_op(ast, metas, prop_feats, op, executor,
                                      base_ms, base_hash,
                                      int(C.TO1_R13_PROBE_PER_ROOT))
        G = max(G1, Gd)
        mem_hit = False
        ms, msucc, mgain = 0.0, 0.0, 0.0
        if G <= 0.0:
            ms, msucc, mgain = memory_raw[op]
            mem_hit = (ms >= float(C.TO1_R13_MEM_SUPPORT_MIN) and
                       msucc >= float(C.TO1_R13_MEM_SUCCESS_MIN) and
                       float(pm.retrieval_gate(iid, episode_id, step, sf)) > 0.0)
        tier = _tier_label(G, mem_hit)
        probe[op] = {"g_direct": G1, "g_dep": Gd, "g_probe": G, "n_execs": nex,
                     "n_pos": npos, "tier": tier, "mem_support": ms,
                     "mem_success": msucc, "mem_mean_gain": mgain}
        tiers[op] = tier
        if G > 0.0:
            pos_probe_cnt += 1
            best_probe_gain = max(best_probe_gain, G)
        if tier == "B":
            mem_rescued += 1

    # Tier A uncapped (never vetoed); Tier B capped by memory gain (B_MEMORY).
    retired = {}
    tier_a = sorted([o for o, t in tiers.items() if t == "A"])
    b_ops = sorted([o for o, t in tiers.items() if t == "B"],
                   key=lambda o: -probe[o]["mem_mean_gain"])
    tier_b = b_ops[: int(C.TO1_R13_MEMORY_BUDGET)]
    retained = sorted(set(tier_a) | set(tier_b))
    tiers_b_final = set(tier_b)

    pool = ast["pool"]
    kept = []
    for k, m in enumerate(metas):
        if m["kind"] == "single":
            op = pool[m["i"]]["e"].operation_id
            if op in retained:
                kept.append(k)
        else:
            oid_a, oid_b = (pool[m["i"]]["e"].operation_id,
                            pool[m["j"]]["e"].operation_id)
            if (oid_a in retained) or (oid_b in retained):
                kept.append(k)
    gated_metas = [metas[k] for k in kept]
    gated_prop = prop_feats[kept] if kept else prop_feats[:0]

    m2_rec = {
        "n_cand": n_cand, "ops": ops,
        "cand_f": cand["feats"].detach().float(),
        "cand_base": cand["base"].detach().float(),
        "cand_latent": cand["latents"].detach().float(),
        "cand_app_raw": cand["app_raw"].detach().float(),
        "cand_app_mask": cand["app_mask"].clone(),
        "cand_trace_graph": cand.get("trace_graph"),
        "cand_min_hop": cand["min_hop"].clone(),
        "cand_support_count": cand["support_count"].clone(),
        "cand_support_blocks": cand["support_blocks"],
        "cand_memory": cand["memory_evidence"].detach().float(),
        "state_context": cand["state_context"].detach().float(),
        "enab_mask": cand["enab_mask"].clone(),
        "anchors": anchors, "draws": draws, "outsider": outsider,
        "pool_mask": pool_mask, "retained": retained,
        "tier_a": tier_a, "tier_b": tiers_b_final, "tiers": dict(tiers),
        "probe": probe,
    }
    diag = {
        "broad_root_count": n_cand,
        "attributed_root_count": int((~cand["enab_mask"]).sum().item()),
        "probed_root_count": len(order),
        "tier_A": len(tier_a), "tier_B": len(tier_b), "tier_C": n_cand - len(retained),
        "positive_probe_count": pos_probe_cnt, "best_probe_gain": best_probe_gain,
        "memory_rescue_count": mem_rescued,
        "gated_proposal_count": len(kept), "full_proposal_count": len(metas),
        "anchor_count": len(anchors), "draw_count": len(draws),
        "outsider_set": outsider >= 0,
    }
    return {"gated_metas": gated_metas, "gated_prop_feats": gated_prop,
            "m2_rec": m2_rec, "diag": diag, "retained": retained,
            "kept_indices": kept}


# ---------------------------------------------------------------------------
# R14 §0 gate: ADAPTIVE probe budget (8 -> 16 -> 24) + per-draw q2 local reward
# ---------------------------------------------------------------------------
def _m2_gate_step_r14(ast, metas, prop_feats, adapter, executor, pm, iid, episode_id,
                      step, sf, rng):
    """The R14 permanent gate (§3, §6-12, §23-28).

    Same makespan-first Tier A/B/C filter as `_m2_gate_step`, with two deltas:

    1. ADAPTIVE probe budget (§23-28): wave 1 = B_INIT (3 anchors + POLICY_DRAWS
       policy draws + 1 outsider).  If tier_A < TIER_A_STOP, expand in B_STEP waves
       (policy draws + 1 high-attribution unseen + 1 outsider) until tier_A >=
       TIER_A_STOP or total = B_MAX.  Memory does NOT control expansion -- Tier-A
       coverage does (§23).  Only POLICY draws carry M2 policy gradient (§12).
    2. q2 local reward per policy draw (§10): PROVEN_GAIN -> 2 + g_norm, MEMORY_RESCUED
       -> 0.5 * memory_confidence(u), else 0.  g_norm = g(u)/max_positive_probe_gain
       (same-state probed roots; single positive -> 1) -- never reads unprobed roots'
       true gain (§11).  memory_confidence(u) = ms/(ms+k) ∈ [0,1) is a bounded
       transform of the fine record COUNT (ms = len(retrieved records), unbounded in
       the real memory), so q2_mem = 0.5·conf is STRUCTURALLY < 0.5 while
       min(proven) = 2+g_norm >= 2 -- the §10/§34 ordering
       max(memory) < min(positive) holds for ALL data and is asserted + recorded in
       diag (§34).

    Matches `_m2_gate_step` output shape (gated_metas / gated_prop_feats / m2_rec /
    diag / retained / kept_indices) so collect/greedy/diag consumers are unchanged
    except the gate they call.
    """
    cand = dict(_root_candidate_arrays(ast))
    cand["state_context"] = torch.as_tensor(sf, dtype=torch.float32).detach()
    n_cand = cand["n_cand"]
    ops = cand["ops"]
    cand["memory_evidence"], memory_raw = _root_memory_features(
        pm, iid, episode_id, step, sf, ast, metas, prop_feats, ops)
    cand["memory_evidence"] = cand["memory_evidence"].detach()
    anchors = []
    for i in range(n_cand):
        if not bool(cand["enab_mask"][i]) and len(anchors) < int(C.TO1_R13_ANCHOR_ROOTS):
            anchors.append(i)
    anchor_set = set(anchors)
    pool_mask = [i for i in range(n_cand) if i not in anchor_set]
    base_ms = int(ast["schedule"].makespan)
    base_hash = schedule_hash(ast["schedule"])

    probe, tiers, order = {}, {}, []
    pos_probe_cnt = 0
    best_probe_gain = 0.0
    mem_rescued = 0

    def _probe_and_tier(i):
        nonlocal pos_probe_cnt, best_probe_gain, mem_rescued
        op = ops[i]
        G1, Gd, nex, npos = _probe_op(ast, metas, prop_feats, op, executor,
                                      base_ms, base_hash,
                                      int(C.TO1_R13_PROBE_PER_ROOT))
        G = max(G1, Gd)
        mem_hit = False
        ms, msucc, mgain = 0.0, 0.0, 0.0
        if G <= 0.0:
            ms, msucc, mgain = memory_raw[op]
            mem_hit = (ms >= float(C.TO1_R13_MEM_SUPPORT_MIN) and
                       msucc >= float(C.TO1_R13_MEM_SUCCESS_MIN) and
                       float(pm.retrieval_gate(iid, episode_id, step, sf)) > 0.0)
        tier = _tier_label(G, mem_hit)
        probe[op] = {"g_direct": G1, "g_dep": Gd, "g_probe": G, "n_execs": nex,
                     "n_pos": npos, "tier": tier, "mem_support": ms,
                     "mem_success": msucc, "mem_mean_gain": mgain}
        tiers[op] = tier
        if G > 0.0:
            pos_probe_cnt += 1
            best_probe_gain = max(best_probe_gain, G)
        if tier == "B":
            mem_rescued += 1
        order.append(i)

    draws_all, outsiders, n_high_attr, waves = [], 0, 0, 0
    # ---- wave 1: B_INIT = 3 anchors + 4 policy + 1 outsider -------------------
    for i in anchors:
        _probe_and_tier(i)
    w1_pol = int(C.TO1_R13_POLICY_DRAWS)
    draws1 = _sequential_draws(adapter, cand, pool_mask, w1_pol, rng)
    drawn = {d["idx"] for d in draws1}
    for d in draws1:
        _probe_and_tier(d["idx"])
    outer_pool = [i for i in pool_mask if i not in drawn]
    if outer_pool:
        _probe_and_tier(int(rng.choice(outer_pool)))
        outsiders += 1
    draws_all += draws1
    waves = 1 if len(order) > 0 else 0
    total = len(order)
    tier_a_cnt = sum(1 for t in tiers.values() if t == "A")

    # ---- expansion waves (§23-28): stop at tier_A >= TIER_A_STOP or B_MAX -----
    probed_set = set(order)
    while tier_a_cnt < int(C.TO1_R14_TIER_A_STOP) and total < int(C.TO1_R14_B_MAX):
        cap = min(int(C.TO1_R14_B_STEP), int(C.TO1_R14_B_MAX) - total)
        remain = [i for i in pool_mask if i not in probed_set]
        if not remain:
            break
        n_pol = max(1, cap - 2)
        new_draws = _sequential_draws(adapter, cand, remain,
                                      min(n_pol, len(remain)), rng)
        nd = {d["idx"] for d in new_draws}
        remain2 = [i for i in remain if i not in nd]
        high_i, outer_i = -1, -1
        if remain2:
            # high-attribution unseen: highest standardized base among remaining
            high_i = int(max(remain2, key=lambda k: float(cand["base"][k])))
            remain3 = [i for i in remain2 if i != high_i]
            if remain3:
                outer_i = int(rng.choice(remain3))
        for d in new_draws:
            _probe_and_tier(d["idx"])
        if high_i >= 0:
            _probe_and_tier(high_i)
            n_high_attr += 1
        if outer_i >= 0:
            _probe_and_tier(outer_i)
            outsiders += 1
        draws_all += new_draws
        waves += 1
        probed_set.update(order)
        total = len(order)
        tier_a_cnt = sum(1 for t in tiers.values() if t == "A")

    # ---- q2 local reward (§10-12): ONLY policy draws train / get reward --------
    q2_stats = {"proven": [], "memory": [], "unsupported": 0}
    for d in draws_all:
        u = ops[d["idx"]]
        g = probe[u]["g_probe"]
        tier = tiers[u]
        if g > 0.0:
            d["g_norm"] = float(g) / max(best_probe_gain, 1e-9)      # §11: <= 1
            d["q2"] = 2.0 + d["g_norm"]
            d["reward_class"] = "proven_gain"
            q2_stats["proven"].append(d["q2"])
        elif tier == "B":
            d["g_norm"] = 0.0
            _ms = float(probe[u]["mem_support"])
            # bounded memory confidence ∈ [0,1): ms/(ms+k); q2_mem = 0.5·conf < 0.5
            d["q2"] = 0.5 * (_ms / (_ms + float(C.TO1_R14_MEM_CONF_SCALE)))
            d["reward_class"] = "memory_rescued"
            q2_stats["memory"].append(d["q2"])
        else:
            d["g_norm"] = 0.0
            d["q2"] = 0.0
            d["reward_class"] = "unsupported"
            q2_stats["unsupported"] += 1
    # §34 structural ordering assertion: min(proven) >= 2 > max(memory) <= 0.5
    min_proven = min(q2_stats["proven"]) if q2_stats["proven"] else None
    max_mem = max(q2_stats["memory"]) if q2_stats["memory"] else None
    ordering_ok = bool(min_proven is None or max_mem is None or min_proven > max_mem)
    assert ordering_ok, "§10 reward ordering PROVEN_GAIN > MEMORY_RESCUED violated"

    # ---- Tier A uncapped; Tier B memory cap; gated Proposal pool (as R13) ------
    tier_a = sorted([o for o, t in tiers.items() if t == "A"])
    b_ops = sorted([o for o, t in tiers.items() if t == "B"],
                   key=lambda o: -probe[o]["mem_mean_gain"])
    tier_b = b_ops[: int(C.TO1_R13_MEMORY_BUDGET)]
    retained = sorted(set(tier_a) | set(tier_b))
    pool = ast["pool"]
    kept = []
    for k, m in enumerate(metas):
        if m["kind"] == "single":
            if pool[m["i"]]["e"].operation_id in retained:
                kept.append(k)
        else:
            oid_a, oid_b = (pool[m["i"]]["e"].operation_id,
                            pool[m["j"]]["e"].operation_id)
            if (oid_a in retained) or (oid_b in retained):
                kept.append(k)
    gated_metas = [metas[k] for k in kept]
    gated_prop = prop_feats[kept] if kept else prop_feats[:0]

    m2_rec = {
        "n_cand": n_cand, "ops": ops,
        "cand_f": cand["feats"].detach().float(),
        "cand_base": cand["base"].detach().float(),
        "cand_latent": cand["latents"].detach().float(),
        "cand_app_raw": cand["app_raw"].detach().float(),
        "cand_app_mask": cand["app_mask"].clone(),
        "cand_trace_graph": cand.get("trace_graph"),
        "cand_min_hop": cand["min_hop"].clone(),
        "cand_support_count": cand["support_count"].clone(),
        "cand_support_blocks": cand["support_blocks"],
        "cand_memory": cand["memory_evidence"].detach().float(),
        "state_context": cand["state_context"].detach().float(),
        "enab_mask": cand["enab_mask"].clone(),
        "anchors": anchors, "draws": draws_all, "outsider": outsiders,
        "pool_mask": pool_mask, "retained": retained,
        "tier_a": tier_a, "tier_b": tier_b, "tiers": dict(tiers),
        "probe": probe, "q2_stats": dict(q2_stats),
    }
    diag = {
        "broad_root_count": n_cand,
        "attributed_root_count": int((~cand["enab_mask"]).sum().item()),
        "probed_root_count": total,
        "tier_A": len(tier_a), "tier_B": len(tier_b), "tier_C": n_cand - len(retained),
        "positive_probe_count": pos_probe_cnt, "best_probe_gain": best_probe_gain,
        "memory_rescue_count": mem_rescued,
        "gated_proposal_count": len(kept), "full_proposal_count": len(metas),
        "anchor_count": len(anchors), "draw_count": len(draws_all),
        "outsider_set": outsiders > 0, "high_attr_count": n_high_attr,
        "expansion_waves": waves,
        "tier_a_stop_reached": bool(tier_a_cnt >= int(C.TO1_R14_TIER_A_STOP)),
        "budget_exhausted": bool(total >= int(C.TO1_R14_B_MAX)),
        "q2_ordering_ok": ordering_ok, "min_proven_q2": min_proven,
        "max_mem_q2": max_mem, "n_unsupported_draws": q2_stats["unsupported"],
    }
    return {"gated_metas": gated_metas, "gated_prop_feats": gated_prop,
            "m2_rec": m2_rec, "diag": diag, "retained": retained,
            "kept_indices": kept}


def _m2_policy_sample_gate(ast, metas, prop_feats, adapter, pm, iid, episode_id,
                           step, sf, rng, *, greedy=False,
                           stratified_quantile=None):
    """Select a small *set* of causal roots without counterfactual replay.

    T2-M v3 removes the single-root hard bottleneck.  At collection time M2 now
    selects up to ``M2_ROOT_TOP_K`` actionable roots.  Training uses exact
    sequential sampling without replacement (so every selected root keeps a
    well-defined conditional old log-prob); greedy evaluation takes sequential
    policy top-K.  A stratified rollout fixes only the first/root-stratum draw and
    samples the remaining roots normally, preserving the existing cross-sibling
    coverage scheme while still expanding several roots.

    Every structurally legal proposal touching *any* selected root reaches M3.
    No G1/GH probe or oracle-derived utility is used here; terminal trajectory
    reward remains the source of M2 credit.
    """
    cand = dict(_root_candidate_arrays(ast))
    from .state_exploration import bias_for
    cand['_explore_bias']=bias_for(getattr(C,'STATE_EXPLORATION',{}),
        schedule_hash(ast['schedule']),'roots',cand['ops'],C.TO1_R13_TEMP_M2)
    cand["state_context"] = torch.as_tensor(sf, dtype=torch.float32).detach()
    n_cand = int(cand["n_cand"])
    ops = cand["ops"]
    cand["memory_evidence"], _memory_raw = _root_memory_features(
        pm, iid, episode_id, step, sf, ast, metas, prop_feats, ops)
    cand["memory_evidence"] = cand["memory_evidence"].detach()
    relation_dim = int(getattr(adapter, "relation_dim", 0))
    cand.setdefault("app_raw", torch.zeros((n_cand, 1, relation_dim)))
    cand.setdefault("app_mask", torch.zeros((n_cand, 1), dtype=torch.bool))
    cand.setdefault("min_hop", torch.full((n_cand,), -1, dtype=torch.int16))
    cand.setdefault("support_count", torch.zeros((n_cand,), dtype=torch.int16))
    cand.setdefault("support_blocks", tuple(() for _ in range(n_cand)))
    if n_cand and hasattr(adapter, "trace_bias"):
        # The reverse graph and state do not change between root-set draws.
        with torch.no_grad():
            cand["_trace_bias"] = adapter.trace_bias(
                cand.get("trace_graph"), cand.get("state_context"))

    # Static legality ownership mask only -- no counterfactual outcome search.
    broad_pool = ast["pool"]
    actionable_ops = set()
    for meta in metas:
        indices = ((meta["i"],) if meta["kind"] == "single"
                   else (meta["i"], meta["j"]))
        for index in indices:
            rec = broad_pool[index]
            actionable_ops.update(rec.get("root_ops") or
                                  (rec["e"].operation_id,))
    pool_mask = [i for i, op in enumerate(ops) if op in actionable_ops]
    n_select = min(max(int(getattr(C, "T2L_M2_ROOT_TOP_K", 1)), 1),
                   len(pool_mask))

    def _policy_probs(rem):
        ids = torch.as_tensor(rem, dtype=torch.long)
        with torch.no_grad():
            if hasattr(adapter, "final_logits_from_candidates"):
                raw = adapter.final_logits_from_candidates(cand, ids)
            else:
                raw = adapter.score(cand["base"][ids], cand["feats"][ids])
            raw=raw+cand['_explore_bias'][ids].to(raw.device)
            probs = torch.softmax(raw / float(C.TO1_R13_TEMP_M2), dim=0)
            eps2 = float(C.TO1_R13_MIX_EPS)
            probs = (1.0 - eps2) * probs + eps2 / max(len(rem), 1)
        return probs, eps2

    draws = []
    if pool_mask and greedy:
        rem = list(pool_mask)
        for _ in range(n_select):
            probs, eps2 = _policy_probs(rem)
            pos = int(probs.argmax().item())
            entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
            draws.append({
                "idx": rem[pos], "rem_ids": tuple(rem),
                "logp": float(torch.log(probs[pos].clamp_min(1e-12)).item()),
                "policy_entropy": entropy,
                "effective_roots": float(np.exp(entropy)),
                "m2_mix_eps": eps2, "selection_mode": "greedy_topk",
            })
            rem.pop(pos)
    elif pool_mask and stratified_quantile is not None:
        # Preserve the old root-stratum semantics for the first draw.  Remaining
        # members are exact on-policy sequential draws from the remaining set.
        probs, eps2 = _policy_probs(pool_mask)
        q = min(max(float(stratified_quantile), 0.0), 1.0 - 1e-12)
        pos = int(torch.searchsorted(
            probs.cumsum(0), torch.tensor(q, dtype=probs.dtype)).item())
        pos = min(pos, len(pool_mask) - 1)
        entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
        first_idx = pool_mask[pos]
        draws = [{
            "idx": first_idx, "rem_ids": tuple(pool_mask),
            "logp": float(torch.log(probs[pos].clamp_min(1e-12)).item()),
            "stratified_quantile": q, "policy_entropy": entropy,
            "effective_roots": float(np.exp(entropy)),
            "m2_mix_eps": eps2, "selection_mode": "stratified_primary",
        }]
        if n_select > 1:
            remaining = [idx for idx in pool_mask if idx != first_idx]
            draws.extend(_sequential_draws(
                adapter, cand, remaining, n_select - 1, rng))
    elif pool_mask:
        draws = _sequential_draws(adapter, cand, pool_mask, n_select, rng)

    selected_indices = [int(draw["idx"]) for draw in draws]
    retained = [ops[idx] for idx in selected_indices]
    selected_idx = selected_indices[0] if selected_indices else -1
    b5_top_idx = (max(pool_mask, key=lambda i: float(cand["base"][i]))
                  if pool_mask else -1)
    policy_diag = None
    if selected_idx >= 0 and hasattr(adapter, "diagnostics_from_candidates"):
        policy_diag = adapter.diagnostics_from_candidates(cand, [selected_idx])
    selected_hop = (int(cand["min_hop"][selected_idx])
                    if selected_idx >= 0 else -1)
    selected_support = (int(cand["support_count"][selected_idx])
                        if selected_idx >= 0 else 0)

    retained_set = set(retained)
    kept = []
    for k, meta in enumerate(metas):
        indices = ((meta["i"],) if meta["kind"] == "single"
                   else (meta["i"], meta["j"]))
        touched = set()
        for index in indices:
            rec = broad_pool[index]
            touched.update(rec.get("root_ops") or
                           (rec["e"].operation_id,))
        if touched & retained_set:
            kept.append(k)
    gated_metas = [metas[k] for k in kept]
    gated_prop = prop_feats[kept] if kept else prop_feats[:0]
    m2_rec = {
        "n_cand": n_cand, "ops": ops,
        "cand_f": cand["feats"].detach().float(),
        "cand_base": cand["base"].detach().float(),
        "cand_latent": cand["latents"].detach().float(),
        "cand_app_raw": cand["app_raw"].detach().float(),
        "cand_app_mask": cand["app_mask"].clone(),
        "cand_trace_graph": cand.get("trace_graph"),
        "cand_min_hop": cand["min_hop"].clone(),
        "cand_support_count": cand["support_count"].clone(),
        "cand_support_blocks": cand["support_blocks"],
        "cand_memory": cand["memory_evidence"].detach().float(),
        "state_context": cand["state_context"].detach().float(),
        "enab_mask": cand["enab_mask"].clone(),
        "anchors": [], "draws": draws, "outsider": -1,
        "pool_mask": pool_mask, "retained": retained,
        "tier_a": [], "tier_b": [], "tiers": {}, "probe": {},
        "credit_source": "unified_net_intervention_return",
        # M2's action is the entire ordered without-replacement root set.
        # The GRPO update therefore uses one joint log-prob / clipped ratio,
        # rather than pretending the K selected roots are K independent actions.
        "selection_semantics": "ordered_without_replacement_root_set",
        "exploration_bias": cand['_explore_bias'].clone(),
    }
    diag = {
        "mode": "score_sample_multi_root_no_probe", "broad_root_count": n_cand,
        "actionable_root_count": len(pool_mask),
        "configured_root_top_k": int(getattr(C, "T2L_M2_ROOT_TOP_K", 1)),
        "selected_root_count": len(retained),
        "selected_root_indices": tuple(selected_indices),
        "selected_roots": tuple(retained),
        "attributed_root_count": int((~cand["enab_mask"]).sum().item()),
        "probed_root_count": 0, "tier_A": 0, "tier_B": 0, "tier_C": 0,
        "positive_probe_count": 0, "best_probe_gain": 0.0,
        "memory_rescue_count": 0, "gated_proposal_count": len(kept),
        "full_proposal_count": len(metas), "anchor_count": 0,
        "draw_count": len(draws), "outsider_set": False,
        "counterfactual_replays": 0,
        # Backward-compatible primary root = first draw/root stratum.
        "selected_root_index": selected_idx,
        "selected_root": (ops[selected_idx] if selected_idx >= 0 else None),
        "selected_min_hop": selected_hop,
        "selected_support_count": selected_support,
        "selected_support_blocks": (cand["support_blocks"][selected_idx]
                                    if selected_idx >= 0 else ()),
        "b5_top1_index": int(b5_top_idx),
        "b5_top1_root": (ops[b5_top_idx] if b5_top_idx >= 0 else None),
        "b5_top1_min_hop": (int(cand["min_hop"][b5_top_idx])
                            if b5_top_idx >= 0 else -1),
        "selected_differs_from_b5_top1": bool(
            selected_idx >= 0 and b5_top_idx >= 0 and selected_idx != b5_top_idx),
        "selected_prior_trust": (float(policy_diag["prior_trust"][0])
                                 if policy_diag is not None else 1.0),
        "selected_root_value": (float(policy_diag["root_value"][0])
                                if policy_diag is not None else 0.0),
        "selected_trace_probability": (
            float(policy_diag["trace_probability"][0])
            if policy_diag is not None else 0.0),
        "appearance_potential_entropy": (
            float(policy_diag["appearance_entropy"])
            if policy_diag is not None else 0.0),
        "active_appearance_count": (
            int(policy_diag["appearance_count"])
            if policy_diag is not None else 0),
        "trace_expected_depth": (
            float(policy_diag["expected_depth"])
            if policy_diag is not None else 0.0),
        "candidate_hops": cand["min_hop"].tolist(),
        "candidate_support_counts": cand["support_count"].tolist(),
        "root_policy_entropy": (float(draws[0].get("policy_entropy", 0.0))
                                if draws else 0.0),
        "root_policy_effective_count": (
            float(draws[0].get("effective_roots", 0.0)) if draws else 0.0),
        "root_candidate_count": len(pool_mask),
        "m2_mix_eps": float(C.TO1_R13_MIX_EPS),
    }
    return {"gated_metas": gated_metas, "gated_prop_feats": gated_prop,
            "m2_rec": m2_rec, "diag": diag, "retained": retained,
            "kept_indices": kept}


# ---------------------------------------------------------------------------
# R15 §5-9: coverage-preserving M3 shortlist (action-set construction)
# ---------------------------------------------------------------------------
def _proposal_structural_family(ast, meta):
    """R15 §8 structural family of a proposal: single ROUTE / single SEQ /
    joint ROUTE+ROUTE / joint ROUTE+SEQ / joint SEQ+SEQ.
    edit_type maps ROUTE->"ROUTE", any SEQ_* / TIMING_SHIFT -> generic "SEQ-like"
    bucket ("SEQ") so the five canonical families cover every pooled proposal.
    Return "single::{kind}" or "pair::{k1}+{k2}" (k1 <= k2, but CC/EE/CE merge)."""
    edits, kind, sig, role, ptype, src, tgt, op_ids = proposal_identity(ast, meta)
    bucket = []
    for e in edits:
        et = getattr(e, "edit_type", "ROUTE")
        bucket.append("ROUTE" if et == "ROUTE" else "SEQ")
    bucket.sort()
    if len(bucket) == 1:
        return f"single::{bucket[0]}"
    return f"pair::{bucket[0]}+{bucket[1]}"


def _proposal_detailed_family(ast, meta):
    """Exact atomic composition used by Joint-Macro V2 diagnostics/quotas."""
    edits, *_ = proposal_identity(ast, meta)
    order = {"ROUTE": 0, "SEQ_SWAP": 1, "SEQ_INSERT": 2}
    kinds = [str(getattr(edit, "edit_type", "UNKNOWN")) for edit in edits]
    kinds.sort(key=lambda kind: (order.get(kind, 99), kind))
    prefix = "single" if len(kinds) == 1 else "pair"
    return f"{prefix}::{'+'.join(kinds)}"


def _bounded_hierarchical_action_pool(ast, gated_metas, gated_prop, gate,
                                      non_improving_streak):
    """Shared train/eval single+pair pool with bounded companion coverage."""
    all_singles = [k for k, meta in enumerate(gated_metas)
                   if meta.get("kind") == "single"]
    single_cap = max(int(C.T2L_SINGLE_ACTIONS), 0)
    ranked_singles = sorted(
        all_singles, key=lambda k: (-old_base_of(gated_prop, k), k))
    # Reserve interpretable coverage before score-based fill. The 1:1:2 split
    # reflects the larger number of meaningful insertion positions without
    # allowing SEQ_INSERT to crowd ROUTE and SEQ_SWAP out of the action set.
    single_quotas = {
        "single::ROUTE": single_cap // 4,
        "single::SEQ_SWAP": single_cap // 4,
        "single::SEQ_INSERT": single_cap - 2 * (single_cap // 4),
    }
    singles = []
    for family, quota_single in single_quotas.items():
        family_rows = [k for k in ranked_singles
                       if _proposal_detailed_family(ast, gated_metas[k]) == family]
        singles.extend(family_rows[:quota_single])
    singles = list(dict.fromkeys(singles))
    if len(singles) < single_cap:
        singles.extend(k for k in ranked_singles if k not in singles)
    singles = singles[:single_cap]
    all_pairs = [k for k, meta in enumerate(gated_metas)
                 if meta.get("kind") == "pair"]
    pair_cap = (int(C.T2L_PLATEAU_PAIR_ACTIONS)
                if non_improving_streak >= int(C.T2L_PLATEAU_STREAK)
                else int(C.T2L_BASE_PAIR_ACTIONS))
    pair_cap = max(pair_cap, 0)
    ranked_pairs = sorted(
        all_pairs, key=lambda k: (-old_base_of(gated_prop, k), k))
    families = (
        "pair::ROUTE+ROUTE", "pair::ROUTE+SEQ_SWAP",
        "pair::ROUTE+SEQ_INSERT", "pair::SEQ_SWAP+SEQ_SWAP",
        "pair::SEQ_SWAP+SEQ_INSERT",
    )
    quota = int(math.ceil(pair_cap / len(families))) if pair_cap else 0
    selected_root = gate.get("diag", {}).get("selected_root")

    def companion_key(k):
        meta = gated_metas[k]
        root_ops, touched_ops = [], []
        for atom_index in (meta["i"], meta["j"]):
            atom = ast["pool"][atom_index]
            root_ops.extend(atom.get("root_ops") or ())
            touched_ops.extend(sorted(_touched_operations(atom["e"])))
        companions = sorted({str(op) for op in root_ops
                             if op != selected_root})
        return tuple(companions or [str(op) for op in touched_ops
                                    if op != selected_root] or ["self"])

    retained_pairs = []
    for family in families:
        rows = [k for k in ranked_pairs
                if _proposal_detailed_family(ast, gated_metas[k]) == family]
        diverse, seen = [], set()
        for k in rows:
            key = companion_key(k)
            if key not in seen:
                diverse.append(k)
                seen.add(key)
        diverse.extend(k for k in rows if k not in diverse)
        retained_pairs.extend(diverse[:quota])
    retained_pairs = list(dict.fromkeys(retained_pairs))
    if len(retained_pairs) < pair_cap:
        retained_pairs.extend(k for k in ranked_pairs if k not in retained_pairs)
    retained_pairs = retained_pairs[:pair_cap]
    return sorted(singles + retained_pairs), pair_cap


def _rl_hierarchical_action_pool(jpol, gated_metas, F_all, sf_t, traj_ctx,
                                 non_improving_streak, rng, *, greedy=False):
    """Bound the M3 action set with the *current RL actor*, not frozen SFT rank.

    Legality/proposal materialisation remains deterministic upstream.  Within that
    legal set, the behavior actor chooses the high-score portion of the shortlist;
    the remaining budget is a uniform policy-independent tail.  Keeping tail
    exploration independent of actor probabilities avoids introducing an extra
    unrecorded stochastic policy stage into the PPO/GRPO likelihood ratio.

    The old SFT head is therefore only a warm-start/prior inside M3, not the hard
    authority that decides which legal proposals the RL actor is allowed to see.
    """
    n = len(gated_metas)
    if n == 0:
        return [], 0, {"mode": "rl_top_plus_uniform_tail", "n": 0}
    all_singles = [k for k, meta in enumerate(gated_metas)
                   if meta.get("kind") == "single"]
    all_pairs = [k for k, meta in enumerate(gated_metas)
                 if meta.get("kind") == "pair"]
    single_cap = min(max(int(C.T2L_SINGLE_ACTIONS), 0), len(all_singles))
    pair_cap_cfg = (int(C.T2L_PLATEAU_PAIR_ACTIONS)
                    if non_improving_streak >= int(C.T2L_PLATEAU_STREAK)
                    else int(C.T2L_BASE_PAIR_ACTIONS))
    pair_cap = min(max(pair_cap_cfg, 0), len(all_pairs))
    pair_mask_all = torch.tensor(
        [meta.get("kind") == "pair" for meta in gated_metas], dtype=torch.bool)
    with torch.no_grad():
        logits_all = jpol.m3.action_logits(
            F_all, sf_t, _pool_stats_from(F_all), traj_ctx=traj_ctx,
            pair_mask=pair_mask_all)[:n].detach().float().cpu()

    top_fraction = min(max(float(getattr(C, "T2L_M3_POOL_TOP_FRACTION", 0.8)),
                           0.0), 1.0)

    def _pick(rows, cap):
        if cap <= 0 or not rows:
            return [], 0, 0
        ranked = sorted(rows, key=lambda k: (-float(logits_all[k]), k))
        if greedy or cap >= len(ranked):
            picked = ranked[:cap]
            return picked, len(picked), 0
        n_top = min(cap, max(1, int(math.ceil(cap * top_fraction))))
        chosen = ranked[:n_top]
        tail = ranked[n_top:]
        n_tail = min(cap - len(chosen), len(tail))
        if n_tail:
            # Uniform tail is deliberately actor-independent exploration.
            extra = rng.sample(tail, n_tail)
            extra.sort(key=lambda k: (-float(logits_all[k]), k))
            chosen.extend(extra)
        return chosen, n_top, n_tail

    singles, s_top, s_tail = _pick(all_singles, single_cap)
    pairs, p_top, p_tail = _pick(all_pairs, pair_cap)
    pool = singles + pairs
    pool.sort(key=lambda k: (-float(logits_all[k]), k))
    info = {
        "mode": "rl_top_plus_uniform_tail",
        "n": n,
        "final": len(pool),
        "single_cap": single_cap,
        "pair_cap": pair_cap,
        "top_fraction": top_fraction,
        "single_top": s_top, "single_tail": s_tail,
        "pair_top": p_top, "pair_tail": p_tail,
    }
    return pool, pair_cap, info


def _shortlist_sig_hash(sigs):
    """R15 §20 per-state shortlist signature hash (sha256 of sorted proposal sigs)."""
    import hashlib
    return hashlib.sha256(("|".join(sorted(sigs))).encode("utf-8")).hexdigest()[:16]


def build_shortlist_r15(ast, gated_metas, gated_prop_feats, sf_t, jpol, scorer, mem,
                        tier_a_ops, tier_b_ops, rng,
                        cap=None, k_global=None, rolex=None):
    """R15 §5-9 coverage-preserving M3 shortlist over the gated proposal pool.

    The shortlist is the M3 action-set construction: it REPLACES the wide_pool stage
    so the selector acts over ≤ M3_SHORTLIST_CAP proposals + STOP.  Construction uses
    ONLY policy-observable signals (§10 -- NO true_U / frozen-future-utility / oracle):
      Source1 GLOBAL     = frozen M3 SFT base score top K_GLOBAL=12
      Source2 per-root   = each retained Tier-A root -> top-2 proposals touching it
                           (Tier-B -> top-1), by frozen score
      Source3 diversity  = structural-family fill (single ROUTE / single SEQ /
                           joint ROUTE+ROUTE / joint ROUTE+SEQ / joint SEQ+SEQ)
                           if size < CAP
    Multi-root proposals may enter via several buckets then signature-dedupe.
    STOP is NEVER inside the shortlist (final action set = shortlist + STOP).

    Returns (idx_list, info) where idx_list indexes into gated_metas (the final M3
    action set, ordered by frozen score desc then deterministic index tiebreak) and
    info carries per-source counts + the §20 signature hash.
    """
    cap = int(cap if cap is not None else C.TO1_R15_SHORTLIST_CAP)
    k_global = int(k_global if k_global is not None else C.TO1_R15_K_GLOBAL)
    N = len(gated_metas)
    empty = {"N": 0, "n_global": 0, "n_roota": 0, "n_rootb": 0, "n_diversity": 0,
             "n_dedup": 0, "final": 0, "cap": cap, "k_global": k_global,
             "signature": _shortlist_sig_hash([])}
    if N == 0:
        return [], empty

    # --- frozen M3 SFT base score per gated proposal (R6 prop head, NO residual) ---
    if rolex is None:
        rolex = _rollex_of(ast, gated_metas, gated_prop_feats, sf_t)
    with torch.no_grad():
        F_all = _rerank_feats_all(scorer, rolex, mem)          # [N, 277]
        base = jpol.m3._base_raw(F_all)[0].detach().float()    # frozen R6 scores [N]
    base_np = base.numpy()
    order_all = sorted(range(N), key=lambda k: (-float(base_np[k]), k))

    sigs = [proposal_identity(ast, m)[2] for m in gated_metas]
    oids = [set(proposal_identity(ast, m)[7]) for m in gated_metas]
    fams = [_proposal_structural_family(ast, m) for m in gated_metas]

    chosen = []
    seen_sig = set()

    def _add(k):
        s = sigs[k]
        if s not in seen_sig:
            seen_sig.add(s)
            chosen.append(k)

    n_global = n_roota = n_rootb = n_diversity = 0

    # 1) GLOBAL top K_GLOBAL by frozen M3 SFT base score
    for k in order_all[:k_global]:
        _add(k)
        n_global += 1

    # 2) per-root quota: Tier-A top-2 / Tier-B top-1 (by frozen score)
    def _cand_for_root(op):
        return [k for k in range(N) if op in oids[k]]

    for op in tier_a_ops:
        cand = sorted(_cand_for_root(op), key=lambda k: -float(base_np[k]))
        for k in cand[: int(C.TO1_R15_TIER_A_QUOTA)]:
            if sigs[k] in seen_sig:
                continue
            _add(k)
            n_roota += 1
    for op in tier_b_ops:
        cand = sorted(_cand_for_root(op), key=lambda k: -float(base_np[k]))
        for k in cand[: int(C.TO1_R15_TIER_B_QUOTA)]:
            if sigs[k] in seen_sig:
                continue
            _add(k)
            n_rootb += 1

    # 3) structural diversity fill if < CAP (round-robin over the 5 families)
    fam_canon = ["single::ROUTE", "single::SEQ", "pair::ROUTE+ROUTE",
                 "pair::ROUTE+SEQ", "pair::SEQ+SEQ"]
    if int(C.TO1_R15_DIVERSITY_FILL) and len(chosen) < cap:
        rest = [k for k in order_all if sigs[k] not in seen_sig]
        parked = {f: [k for k in rest if fams[k] == f] for f in fam_canon}
        for f in parked:
            parked[f] = sorted(parked[f], key=lambda k: -float(base_np[k]))
        while len(chosen) < cap and any(parked.values()):
            progressed = False
            for f in fam_canon:
                lst = parked[f]
                while lst:
                    k = lst.pop(0)
                    if sigs[k] in seen_sig:
                        continue
                    _add(k)
                    n_diversity += 1
                    progressed = True
                    break
                if len(chosen) >= cap:
                    break
            if not progressed:
                # no family has an unseen member left -> stop filling
                break

    # cap is a hard ceiling (never exceed)
    chosen = chosen[:cap]
    final_sigs = [sigs[k] for k in chosen]
    info = {"N": N, "n_global": n_global, "n_roota": n_roota, "n_rootb": n_rootb,
            "n_diversity": n_diversity, "n_dedup": N - len(chosen),
            "final": len(chosen), "cap": cap, "k_global": k_global,
            "signature": _shortlist_sig_hash(final_sigs),
            "full_broad_roots": len(tier_a_ops) + len(tier_b_ops),
            "tier_a_roots": len(tier_a_ops), "tier_b_roots": len(tier_b_ops),
            "fams": {f: fams.count(f) for f in fam_canon if f in fams}}
    return chosen, info


# ---------------------------------------------------------------------------
# sibling trajectory collection -- R12 semantics + the M2 gate  (§0,9-20,22-24)
# ---------------------------------------------------------------------------
def collect_trajectory_r13(jpol, scorer, executor, model_b5, single_head, direct_head,
                           problem, schedule_root, root_ms, iid, episode_id, progmem_root,
                           seed, traj_id, T=None, eps=None, horizon=None, step_offset=0,
                           seed_rolex=None, stop_on_negative=True, seed_ast=None):
    """One sibling trajectory from a FROZEN root through the §0 pipeline.

    Identical contract to R12 collect_trajectory_r12 (true-STOP semantics: policy
    STOP / infeasible / non-positive / no-proposal / revisit all end the episode;
    only executed steps with improvement > 0 advance; sibling isolation via deep-copy;
    deterministic in `seed`) with ONE inserted stage: after the online
    Appearance->analyze_state->proposal build, the permanent M2 budgeted root probe +
    makespan-first filter + memory rescue gates the proposal pool BEFORE the Reasoner
    view (rolex) reaches the M3 policy. Root probe gains are REAL FDR replays only and
    never feed the terminal reward (§24). M2 draws are recorded per step for the joint
    GRPO ratio (§15).
    """
    T = float(T if T is not None else C.TO1_R13_TEMP)
    eps = float(eps if eps is not None else C.TO1_R13_MIX_EPS)
    horizon = int(horizon if horizon is not None else C.TO1_R13_HORIZON)

    t_prof = {"analyze": 0.0, "execute": 0.0, "mem": 0.0, "policy": 0.0, "m2gate": 0.0}
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
    last_step_gain = 0.0
    non_improving_streak = 0
    for t in range(horizon):
        t1 = time.time()
        if t == 0 and seed_rolex is not None:
            prop_feats, metas, agg = seed_rolex
        else:
            prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
        ast = (seed_ast if t == 0 and seed_ast is not None
               else cache.ast(problem, schedule, iid))
        n_prop = len(metas)
        t_prof["analyze"] += time.time() - t1
        if n_prop == 0:
            terminal = "no_proposals"
            break
        h = schedule_hash(schedule)
        sf = state_feature_vec(ms_cur, root_ms, n_prop, agg["best_uhat"],
                               agg["best_direct"], agg["n_contrib"], agg["n_enab"])
        sf_t = torch.tensor(sf, dtype=torch.float32)
        # ---- M2 budget + probe + makespan-first gate (§0 pipeline) ----------
        t1 = time.time()
        gate = _m2_gate_step(ast, metas, prop_feats, jpol.m2, executor, pm, iid,
                             episode_id, step_offset + t, sf, rng)
        ts = time.time() - t1
        t_prof["m2gate"] += ts
        gated_metas, gated_prop = gate["gated_metas"], gate["gated_prop_feats"]
        if not gated_metas:
            terminal = "no_pool_m2"
            break
        rolex = _rollex_of(ast, gated_metas, gated_prop, sf_t)
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
            terminal = "no_pool_wide"
            break
        mem_sel = mem * gmem
        F_pool = _rerank_feats_all(scorer, rolex, mem_sel)[pool]
        pool_stats = _pool_stats_from(F_pool)
        t1 = time.time()
        with torch.no_grad():
            logits = jpol.m3.action_logits(F_pool, sf_t, pool_stats)
        t_prof["policy"] += time.time() - t1
        a = mixture_sample(logits, T, eps, rng)
        logp_old = float(mixture_logp(logits, a, T, eps))
        M = len(pool)
        is_stop = bool(a == M)
        sig = None if is_stop else proposal_identity(ast, gated_metas[pool[a]])[2]
        if mem_probe is None:
            mem_probe = [float(v) for v in
                         torch.tensor(pm.features(iid, episode_id, step, sf,
                                                  queries[: min(2, len(queries))]))
                         .flatten().tolist()]
        rec = {
            "instance_id": iid, "episode_id": episode_id, "root_state_hash": h0,
            "trajectory_id": traj_id, "step_idx": step, "state_hash": h,
            "action_signature": ("STOP" if is_stop else sig), "is_stop": is_stop,
            "state_makespan_before": int(ms_cur),
            "old_logprob": logp_old, "logp_old": logp_old,
            "M": M, "a": int(a),
            "F_pool": F_pool.detach().float(), "sf_t": sf_t.detach().float(),
            "pool_stats": pool_stats.detach().float(),
            "logits_old": logits.detach().float(),
            "m2_rec": gate["m2_rec"], "m2_diag": gate["diag"],
            "traj": traj_id,
        }
        steps.append(rec)
        if is_stop:
            terminal = "stop"
            break
        meta = gated_metas[pool[a]]
        edits, kind = _edits_for(ast, meta)
        rec["edit_types"] = tuple(str(e.edit_type) for e in edits)
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
            "m2_selected_root": gate.get("diag", {}).get("selected_root"),
            "m2_root_min_hop": gate.get("diag", {}).get("selected_min_hop", -1),
            "m2_root_support_count": gate.get("diag", {}).get(
                "selected_support_count", 0),
        })
        if res is None:
            terminal = "infeasible"
            break
        rec["successor_makespan"] = int(res["schedule"].makespan)
        rec["successor_state_hash"] = schedule_hash(res["schedule"])
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
    # ---- exact terminal reward (§30; probe gain NEVER enters it) -----------
    reward = int(root_ms) - ms_cur
    for rec in steps:
        rec["reward_terminal"] = reward
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
        "prof": t_prof, "semantics": "true_stop", "joint": True,
    }


# ---------------------------------------------------------------------------
# full group rollout -- ProcessPoolExecutor over the §0 pipeline  (§37-41)
# ---------------------------------------------------------------------------
def _mp_worker_init_r13(jpol, scorer, executor, model_b5, single_head, direct_head):
    load_upstream()
    global _MP13_STATE
    _MP13_STATE = {
        "jpol": jpol, "scorer": scorer, "executor": executor,
        "model_b5": model_b5, "single_head": single_head, "direct_head": direct_head,
    }


def _mp_traj_job_r13(contract):
    st = _MP13_STATE
    return collect_trajectory_r13(
        st["jpol"], st["scorer"], st["executor"],
        st["model_b5"], st["single_head"], st["direct_head"],
        contract["problem"], contract["schedule_root"], contract["root_ms"],
        contract["iid"], contract["episode_id"], contract["progmem_root"],
        seed=contract["seed"], traj_id=contract["traj_id"],
        T=contract.get("T"), eps=contract.get("eps"), horizon=contract.get("horizon"),
        step_offset=contract.get("step_offset", 0), seed_rolex=contract.get("seed_rolex"),
        seed_ast=contract.get("seed_ast"))


def _seed_rolex13(cache, problem, schedule, iid):
    """Parent-side precompute of the t=0 FULL pool (the gate filters it inside the
    worker; sibling RNG still decides draws, so worker=1 == worker=N holds)."""
    return cache.proposals(problem, schedule, iid)


def _seed_state13(cache, problem, schedule, iid):
    """Materialize the deterministic t=0 proposal tuple and AST exactly once."""
    seed_rolex = cache.proposals(problem, schedule, iid)
    seed_ast = cache.ast(problem, schedule, iid)
    _root_candidate_arrays(seed_ast)  # finish the only lazy AST-derived cache
    return seed_rolex, seed_ast, cache.ast_seconds_for(schedule, iid)


def collect_full_group_rollouts_r13(jpol, scorer, executor, model_b5, single_head,
                                    direct_head, problem, schedule, root_ms, iid,
                                    episode_id, progmem, k=None, T=None, eps=None,
                                    horizon=None, seed=0, step_offset=0, workers=1,
                                    step0_cache=None, mp_ctx=None):
    """K sibling trajectories from one frozen root through the §0 pipeline.
    workers=1 == workers=N bit-identical (same deterministic collect)."""
    from concurrent.futures import ProcessPoolExecutor
    k = int(k if k is not None else C.TO1_R13_K)
    h0 = schedule_hash(schedule)
    seed_rolex = seed_ast = None
    seed_ast_s = 0.0
    if step0_cache is not None:
        seed_rolex, seed_ast, seed_ast_s = _seed_state13(
            step0_cache, problem, schedule, iid)

    def _contract(kid):
        return {
            "seed": _traj_seed(seed, iid, episode_id, h0, kid), "traj_id": kid,
            "problem": problem, "schedule_root": schedule, "root_ms": int(root_ms),
            "iid": iid, "episode_id": episode_id, "progmem_root": progmem,
            "T": T, "eps": eps, "horizon": horizon, "step_offset": step_offset,
            "seed_rolex": seed_rolex,
            "seed_ast": seed_ast,
        }

    t0 = time.time()
    trajs = []
    if workers and int(workers) > 1:
        ctx = _resolve_mp_ctx(mp_ctx)
        with ProcessPoolExecutor(
                max_workers=int(workers), mp_context=ctx,
                initializer=_mp_worker_init_r13,
                initargs=(jpol, scorer, executor, model_b5, single_head, direct_head)) as ex:
            futs = [ex.submit(_mp_traj_job_r13, _contract(kid)) for kid in range(k)]
            for f in futs:
                trajs.append(f.result())
    else:
        for kid in range(k):
            c = _contract(kid)
            trajs.append(collect_trajectory_r13(
                jpol, scorer, executor, model_b5, single_head, direct_head,
                c["problem"], c["schedule_root"], c["root_ms"], c["iid"], c["episode_id"],
                c["progmem_root"], seed=c["seed"], traj_id=c["traj_id"],
                T=T, eps=eps, horizon=horizon, step_offset=step_offset,
                seed_rolex=seed_rolex, seed_ast=seed_ast))
    coll_s = time.time() - t0

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
        "rewards": [float(x) for x in np.asarray(
            [tr["reward"] for tr in trajs], dtype=np.float64).tolist()],
        "mean_reward": advm["mean"], "std_reward": advm["std"],
        "informative": advm["informative"], "advantages": advm["advantages"],
        "trajs": trajs, "n_steps": sum(tr["n_steps"] for tr in trajs),
        "n_acted": sum(tr["n_acted"] for tr in trajs),
        "terminal_counts": dict(Counter(tr["terminal"] for tr in trajs)),
        "coll_s": coll_s, "mem_unchanged": mem_unchanged,
        "parallelism": ("mp" if (workers and int(workers) > 1) else "serial"),
        "workers": int(workers),
        "ast_profile": {"ast_calls_before": int(k), "ast_calls_after": 1,
                        "ast_seconds_per_call": float(seed_ast_s),
                        "ast_seconds_saved": float(seed_ast_s * max(k - 1, 0))},
    }


# ---------------------------------------------------------------------------
# R14 sibling trajectory collect: ADAPTIVE gate + per-draw q2 + U2 stage utility
# ---------------------------------------------------------------------------
def collect_trajectory_r14(jpol, scorer, executor, model_b5, single_head, direct_head,
                           problem, schedule_root, root_ms, iid, episode_id, progmem_root,
                           seed, traj_id, T=None, eps=None, horizon=None, step_offset=0,
                           seed_rolex=None, stop_on_negative=True, action_space="full",
                           val_cache=None, seed_ast=None, analyze_cache=None,
                           allow_policy_stop=True, feasible_fallback=False,
                           capture_visual_plan=False, anchor_trajectories=0,
                           initial_root_quantile=None, root_group_id=None,
                           initial_action_stratum=None, replay_cache=None):
    """One sibling trajectory under the R14 stage-wise reward regime.

    Identical §0 pipeline to `collect_trajectory_r13`, with two deltas:
    1. the M2 gate runs the ADAPTIVE probe + writes the per-draw q2 local reward
       (`_m2_gate_step_r14`);
    2. the trajectory accumulates the M2 stage utility U2 (§13) = mean over its
       steps of U2_i,t, where U2_i,t = mean over the state's policy draws of q2.
       M3 keeps the terminal makespan reward (§16); probe gains NEVER enter any
       reward (§24, m2_reward_authority=false).  U2 feeds A2 in the group rollout;
       the terminal reward feeds A3 (§14, §17).
    """
    T = float(T if T is not None else C.TO1_R13_TEMP)
    eps = float(eps if eps is not None else C.TO1_R13_MIX_EPS)
    horizon = int(horizon if horizon is not None else C.TO1_R13_HORIZON)

    t_prof = {"analyze": 0.0, "proposal_structure": 0.0,
              "m2gate": 0.0, "mem": 0.0, "m3_features": 0.0,
              "policy": 0.0, "execute": 0.0, "validation": 0.0}
    t0 = time.time()
    schedule = copy.deepcopy(schedule_root)
    pm = copy.deepcopy(progmem_root)
    root_exec_records = copy.deepcopy(pm.executed.get(iid, {}))
    h0 = schedule_hash(schedule_root)
    assert h0 == schedule_hash(schedule), "sibling state isolation violated at copy"
    mem_probe = None
    # A shard's sibling trajectories are isolated in schedule/Memory, while
    # deterministic graph analysis depends only on (problem, schedule, iid).
    # Sharing this read-through cache inside one bounded shard avoids rebuilding
    # identical states without sharing any branch-local mutable state.
    cache = (analyze_cache if analyze_cache is not None else
             AnalyzeCache(model_b5, single_head, direct_head))
    rng = random.Random(seed)
    visited = {h0}
    val_cache = {} if val_cache is None else val_cache
    replay_cache = replay_cache if replay_cache is not None else _new_replay_cache()
    replay_start = (replay_cache.lookups, replay_cache.hits, replay_cache.executions)
    vprobe_n = vprobe_ms = 0                           # §32 validation wall-clock

    ms_cur = int(root_ms)
    # T2-H still optimizes the terminal H-step outcome, but also retains the
    # best prefix reached inside the rollout.  These values must be initialized
    # in R14: T2-H collection calls this function, not the legacy R13 collector.
    best_ms = int(root_ms)
    best_step = 0
    best_schedule = copy.deepcopy(schedule)
    best_memory_records = []
    steps = []
    terminal = None
    n_acted = 0
    step_u2 = []
    last_step_gain = 0.0
    non_improving_streak = 0
    post_m2_root_gate_bypass_attempts = 0
    pval_nonfinite_g1_count = 0
    pval_nonfinite_gh_count = 0
    feasibility_replay_count = 0
    m2_breakdown = {"proven": [], "memory": [], "unsupported": 0, "steps": 0}
    # Optional paper/debug payload.  This records only the selected edit objects,
    # never whole schedule snapshots, so tracing one root does not inflate the
    # already expensive rollout IPC.  The parent replays the best sibling once
    # and renders its exact intermediate schedules.
    visual_plan = [] if bool(capture_visual_plan) else None
    state_sample = None
    for t in range(horizon):
        _worker_heartbeat("analyze_start", iid=iid, traj_id=traj_id, step=t,
                          makespan=int(ms_cur))
        t1 = time.time()
        if t == 0 and seed_rolex is not None:
            prop_feats, metas, agg = seed_rolex
        else:
            prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
        ast = (seed_ast if t == 0 and seed_ast is not None
               else cache.ast(problem, schedule, iid))
        n_prop = len(metas)
        t_prof["analyze"] += time.time() - t1
        _worker_heartbeat("analyze_end", iid=iid, traj_id=traj_id, step=t,
                          proposals=int(n_prop))
        if n_prop == 0:
            terminal = "no_proposals"
            break
        h = schedule_hash(schedule)
        sf = state_feature_vec(ms_cur, root_ms, n_prop, agg["best_uhat"],
                               agg["best_direct"], agg["n_contrib"], agg["n_enab"])
        sf_t = torch.tensor(sf, dtype=torch.float32)
        # ---- M2 adaptive probe + q2 + makespan-first gate (§0, §6-12) ------
        _worker_heartbeat("m2gate_start", iid=iid, traj_id=traj_id, step=t,
                          proposals=int(n_prop))
        t1 = time.time()
        if action_space == "policy_sampled":
            gate = _m2_policy_sample_gate(
                ast, metas, prop_feats, jpol.m2, pm, iid, episode_id,
                step_offset + t, sf, rng, greedy=False,
                stratified_quantile=(initial_root_quantile if t == 0 else None))
        else:
            gate = _m2_gate_step_r14(ast, metas, prop_feats, jpol.m2, executor, pm, iid,
                                     episode_id, step_offset + t, sf, rng)
        ts = time.time() - t1
        t_prof["m2gate"] += ts
        _worker_heartbeat("m2gate_end", iid=iid, traj_id=traj_id, step=t,
                          gated=int(len(gate.get("gated_metas", ()))))
        dq = [float(d.get("q2", 0.0)) for d in gate["m2_rec"]["draws"]]
        step_u2.append(float(np.mean(dq)) if dq else 0.0)
        m2_breakdown["steps"] += 1
        for d in gate["m2_rec"]["draws"]:
            rc = d.get("reward_class", "unsupported")
            if rc == "proven_gain":
                m2_breakdown["proven"].append(float(d["q2"]))
            elif rc == "memory_rescued":
                m2_breakdown["memory"].append(float(d["q2"]))
            else:
                m2_breakdown["unsupported"] += 1
        # T2-F's "continue while a feasible move exists" contract has to apply
        # before M3 as well as to M3's STOP logit.  M2 attribution may be
        # non-empty while its downstream real-probe/Tier-A/B root gate retains
        # nothing, even though the broad structural proposal set still contains
        # executable moves.  In that one case only,
        # let R20 replay the broad set; its feasible fallback then keeps at most
        # four least-damaging moves.  Normal non-empty M2 decisions are unchanged.
        if (action_space != "policy_sampled" and feasible_fallback and
                not gate["gated_metas"] and metas):
            gate = dict(gate)
            gate["gated_metas"] = metas
            gate["gated_prop_feats"] = prop_feats
            gate["diag"] = dict(gate["diag"])
            gate["diag"]["post_m2_root_gate_bypass"] = True
            post_m2_root_gate_bypass_attempts += 1
        else:
            gate["diag"]["post_m2_root_gate_bypass"] = False
        gated_metas, gated_prop = gate["gated_metas"], gate["gated_prop_feats"]
        if not gated_metas:
            terminal = "no_pool_m2"
            break
        t1 = time.time()
        rolex = _rollex_of(ast, gated_metas, gated_prop, sf_t)
        queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
                   for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                             rolex["src"], rolex["tgt"])]
        t_prof["proposal_structure"] += time.time() - t1
        t1 = time.time()
        step = step_offset + t
        mem = torch.tensor(pm.features(iid, episode_id, step, sf, queries),
                           dtype=torch.float32)
        gmem = float(pm.retrieval_gate(iid, episode_id, step, sf))
        t_prof["mem"] += time.time() - t1
        mem_sel = mem * gmem
        sl_info = None
        pool_builder_info = None
        evid = None
        vp_sig = None
        pv_diag = None
        pool_validated_sigs = None
        # The trajectory context is policy-observable and is also needed by the
        # RL-led shortlist builder below.
        traj_ctx = trajectory_context(
            step_index=t, horizon=horizon, root_makespan=root_ms,
            current_makespan=ms_cur, last_step_gain=last_step_gain,
            action_count=n_acted, non_improving_streak=non_improving_streak)
        if action_space == "policy_sampled":
            # T2-M v3: legality/materialisation stays deterministic, but shortlist
            # membership is governed by the current RL actor rather than a frozen
            # SFT ranking.  A uniform tail keeps low-ranked legal moves discoverable.
            t1 = time.time()
            F_all = _rerank_feats_all(scorer, rolex, mem_sel)
            pool, pair_cap, pool_builder_info = _rl_hierarchical_action_pool(
                jpol, gated_metas, F_all, sf_t, traj_ctx,
                non_improving_streak, rng, greedy=False)
            if not pool:
                terminal = "no_policy_sampled_pool"
                break
            F_pool = F_all[pool]
            pool_stats = _pool_stats_from(F_pool)
            pair_mask = torch.tensor([
                gated_metas[k].get("kind") == "pair" for k in pool
            ], dtype=torch.bool)
            t_prof["m3_features"] += time.time() - t1
            logit_pos = rank = None
        elif action_space == "shortlist":
            # R15 §5-9: the M3 action set = coverage-preserving shortlist + STOP.
            # `a` indexes the SHORTLIST (0..M-1); STOP is index M.  `pool` becomes
            # the shortlist index list so all downstream meta lookups stay identical.
            shortlist_idx, sl_info = build_shortlist_r15(
                ast, gated_metas, gated_prop, sf_t, jpol, scorer, mem,
                gate["m2_rec"]["tier_a"], gate["m2_rec"]["tier_b"], rng,
                cap=C.TO1_R15_SHORTLIST_CAP, k_global=C.TO1_R15_K_GLOBAL,
                rolex=rolex)
            if not shortlist_idx:
                terminal = "no_pool_shortlist"
                break
            pool = shortlist_idx
            logit_pos = rank = None
        elif action_space == "validated":
            # R18 §4-13: REAL per-Proposal counterfactual validation (G_prop) then
            # makespan-first / Memory-second Proposal filtering.  `pool` = validated
            # proposal indices into `gated_metas`; STOP is index M.  NO cap32, NO
            # extra Top-K.  ProposalProbeGain enters ONLY the observation `evid`.
            t1 = time.time()
            vs = build_validated_action_set_r18(
                ast, gate, rolex, scorer, executor, pm, iid, episode_id,
                step_offset + t, sf, h, ms_cur, mem_sel, cache=val_cache)
            vprobe_ms += time.time() - t1
            vprobe_n += 1
            if not vs["pool"]:
                terminal = "no_validated_pool"
                break
            pool = vs["pool"]
            F_pool = vs["F_pool"]
            pool_stats = vs["pool_stats"]
            evid = vs["evid"]
            vp_sig = vs["validated_pool_signature"]
            pool_validated_sigs = vs["signatures"]
            pv_diag = vs["pval"]["diag"]
            logit_pos = rank = None
        elif action_space == "multistep":
            # R19 §6-24: H-step-validated action set -- ALL IMMEDIATE_POSITIVE ∪
            # ALL DELAYED_POSITIVE ∪ MAX-4 MEMORY_RESCUED ∪ STOP (NO cap32 §21).
            # The bounded continuation makes trajectory-visible gains appear at
            # S_t, so the pool no longer loses them to myopic G_prop filtering
            # (R18 verdict D: delayed_benefit=0.596).  `acache` = THIS loop's
            # AnalyzeCache for branch-state pool re-derivation (§40); G1/GH are
            # observation-only `evid`, NEVER rewards (§33).
            t1 = time.time()
            vs = build_multistep_action_set_r19(
                ast, gate, rolex, scorer, executor, pm, iid, episode_id,
                step_offset + t, sf, h, ms_cur, mem_sel,
                cont_m2=getattr(jpol, "cont_m2", None),
                cont_m3=getattr(jpol, "cont_m3", None),
                acache=cache, rc=val_cache, hcache=val_cache)
            vprobe_ms += time.time() - t1
            vprobe_n += 1
            if not vs["pool"]:
                terminal = "no_multistep_pool"
                break
            pool = vs["pool"]
            F_pool = vs["F_pool"]
            pool_stats = vs["pool_stats"]
            evid = vs["evid"]
            vp_sig = vs["pool_sig_hash"]
            pool_validated_sigs = vs["signatures"]
            pv_diag = vs["pval"]["diag"]
            logit_pos = rank = None
        elif action_space == "lexicographic":
            # R20 §4-12: state-level lexicographic fallback (PA >> PB >> PC >> STOP).
            # GH continuation runs ONLY when PA_t is empty (§5/§15) -> fewer calls.
            t1 = time.time()
            vs = build_lexicographic_action_set_r20(
                ast, gate, rolex, scorer, executor, pm, iid, episode_id,
                step_offset + t, sf, h, ms_cur, mem_sel,
                cont_m2=getattr(jpol, "cont_m2", None),
                cont_m3=getattr(jpol, "cont_m3", None),
                acache=cache, rc=val_cache, hcache=val_cache,
                feasible_fallback=bool(feasible_fallback))
            vprobe_ms += time.time() - t1
            vprobe_n += 1
            pval_nonfinite_g1_count += int(
                vs["pval"]["diag"].get("nonfinite_g1_count", 0))
            pval_nonfinite_gh_count += int(
                vs["pval"]["diag"].get("nonfinite_gh_count", 0))
            if not vs["pool"]:
                terminal = "no_lexicographic_pool"
                break
            pool = vs["pool"]
            F_pool = vs["F_pool"]
            pool_stats = vs["pool_stats"]
            evid = vs["evid"]
            vp_sig = vs["pool_sig_hash"]
            pool_validated_sigs = vs["signatures"]
            pv_diag = vs["pval"]["diag"]
            logit_pos = rank = None
        else:
            logit_pos, rank = _scores(scorer, rolex, mem)
            pool, pinfo = wide_pool(rolex, logit_pos, rank)
            if not pool:
                terminal = "no_pool_wide"
                break
            F_pool = _rerank_feats_all(scorer, rolex, mem_sel)[pool]
            pool_stats = _pool_stats_from(F_pool)
        t1 = time.time()
        if action_space != "policy_sampled":
            pair_mask = None
        with torch.no_grad():
            logits = jpol.m3.action_logits(F_pool, sf_t, pool_stats, evid=evid,
                                           traj_ctx=traj_ctx,
                                           pair_mask=pair_mask)
        t_prof["policy"] += time.time() - t1
        # A small configurable subset of siblings are cheap guided anchors.
        # Anchors inspect only a bounded actor-ranked prefix for feasibility, so
        # this never restores the expensive all-candidate G1/GH/R20 pipeline.
        # Anchors are excluded from group-relative GRPO below and contribute only
        # a 0.05-weight successful-prefix self-imitation signal.
        anchor_mode = (action_space == "policy_sampled" and
                       isinstance(traj_id, int) and
                       0 <= traj_id < int(anchor_trajectories))
        anchor_successor = None
        anchor_choice = None
        if anchor_mode:
            order = torch.argsort(logits[:len(pool)], descending=True).tolist()
            feasible = []
            t1 = time.time()
            for old_index in order[:min(
                    max(1, int(getattr(C, "T2L_ANCHOR_SCAN", 4))), len(order))]:
                old_meta = gated_metas[pool[old_index]]
                old_edits, _old_kind = _edits_for(ast, old_meta)
                checked, _reason, _hit = replay_cache.execute(
                    _execute_step_with_reason, executor, problem, schedule, old_edits, ms_cur, h)
                feasibility_replay_count += 1
                if checked is not None:
                    feasible.append((old_index, checked))
                    if len(feasible) >= max(
                            1, int(getattr(C, "T2L_ANCHOR_KEEP", 2))):
                        break
            t_prof["validation"] += time.time() - t1
            if not feasible:
                terminal = "no_executable_policy_candidates"
                break
            feasible.sort(key=lambda item: (-float(item[1]["improvement"]),
                                            int(item[0])))
            anchor_choice, anchor_successor = feasible[int(traj_id) % len(feasible)]
        M = len(pool)
        # Persistent-frontier training must spend its fixed H-step search budget.
        # STOP remains in the frozen parent head for architectural/checkpoint
        # compatibility, but is masked whenever a validated executable proposal
        # exists.  Empty pools/replay failure/revisits remain safety terminals.
        sample_logits = logits if bool(allow_policy_stop) else logits[:M]
        from .state_exploration import bias_for
        m3_exploration_bias=bias_for(getattr(C,'STATE_EXPLORATION',{}),h,'actions',
            [proposal_identity(ast,gated_metas[k])[2] for k in pool],T)
        if bool(allow_policy_stop):
            m3_exploration_bias=torch.cat([m3_exploration_bias,torch.zeros(1)])
        sample_logits=sample_logits+m3_exploration_bias.to(sample_logits.device)
        pair_probability_mass = 0.0
        pair_candidate_rate = 0.0
        if pair_mask is not None and M:
            behavior_p = mixture_pmf(sample_logits, T, eps)
            pair_probability_mass = float(
                behavior_p[:M][pair_mask].sum().detach().cpu())
            pair_candidate_rate = float(pair_mask.float().mean())
        m3_context_key = None
        m3_quantile = None
        m3_sampling_mode = "anchor" if anchor_mode else "independent"
        if (t == 0 and action_space == "policy_sampled" and not anchor_mode
                and initial_action_stratum is not None
                and bool(C.T2L_M3_FIRST_STEP_STRATIFIED)):
            # Same complete actor-visible context -> shared independent rotation.
            # Different Memory/features/pool order -> different context fingerprint.
            identities = [str(proposal_identity(ast, gated_metas[idx])[2]) for idx in pool]
            m3_context_key = policy_context_fingerprint(
                {"iid": str(iid), "state": h, "episode": str(episode_id),
                 "step": int(step), "actions": identities,
                 "stop_allowed": bool(allow_policy_stop), "T": T, "eps": eps},
                {"features": F_pool, "state": sf_t, "stats": pool_stats,
                 "evidence": evid, "trajectory": traj_ctx, "pair_mask": pair_mask,
                 "memory": mem_sel, "logits": sample_logits})
            a, m3_quantile = rotated_action_sample(
                mixture_pmf(sample_logits, T, eps), initial_action_stratum, m3_context_key)
            m3_sampling_mode = "rotated_first_step"
        else:
            a = (int(anchor_choice) if anchor_choice is not None else
                 mixture_sample(sample_logits, T, eps, rng))
        logp_old = float(mixture_logp(sample_logits, a, T, eps))
        is_stop = bool(a == M)
        sig = None if is_stop else proposal_identity(ast, gated_metas[pool[a]])[2]
        if mem_probe is None:
            mem_probe = [float(v) for v in
                         torch.tensor(pm.features(iid, episode_id, step, sf,
                                                  queries[: min(2, len(queries))]))
                         .flatten().tolist()]
        rec = {
            "instance_id": iid, "episode_id": episode_id, "root_state_hash": h0,
            "trajectory_id": traj_id, "step_idx": step, "state_hash": h,
            "action_signature": ("STOP" if is_stop else sig), "is_stop": is_stop,
            "state_makespan_before": int(ms_cur),
            "old_logprob": logp_old, "logp_old": logp_old,
            "M": M, "a": int(a),
            "stop_allowed": bool(allow_policy_stop),
            "pair_mask": (pair_mask.detach().clone()
                          if pair_mask is not None else None),
            "pair_candidate_rate": pair_candidate_rate,
            "pair_probability_mass": pair_probability_mass,
            "selected_is_pair": bool(
                pair_mask is not None and not is_stop and bool(pair_mask[a])),
            "F_pool": F_pool.detach().float(), "sf_t": sf_t.detach().float(),
            "pool_stats": pool_stats.detach().float(),
            "logits_old": logits.detach().float(),
            "m2_rec": gate["m2_rec"], "m2_diag": gate["diag"],
            "traj": traj_id, "U2_t": step_u2[-1],
            "action_space": action_space,
            "m3_sampling_mode": m3_sampling_mode,
            "m3_exploration_bias": m3_exploration_bias[:M].clone(),
            "m3_first_context_key": m3_context_key,
            "m3_first_quantile": m3_quantile,
            "is_anchor": bool(anchor_mode),
            "evid": (evid.detach().float() if evid is not None else None),
            "traj_ctx": traj_ctx.detach().float(),
            "validated_pool_signature": vp_sig,
            "pool_validated_sigs": pool_validated_sigs,
            "pv_diag": pv_diag,
            "sampled_class": (None if action_space not in ("multistep", "lexicographic") or is_stop else
                              next(r["class"] for r in vs["pval"]["rows"]
                                   if r["k"] == pool[a])),
            "active_layer": (None if action_space != "lexicographic" else
                             vs["pval"]["diag"].get("active_layer")),
            "sl_info": (sl_info if sl_info is not None else
                        {"signature": None, "final": M, "N": M, "cap": M,
                         "k_global": 0, "n_global": 0, "n_roota": 0,
                         "n_rootb": 0, "n_diversity": 0, "n_dedup": 0}),
            "pool_builder_info": pool_builder_info,
        }
        if getattr(model_b5, "e2e_capture", False):
            from .end_to_end import capture_decision
            rec["e2e"] = capture_decision(ast, gated_metas, pool, mem_sel, gated_prop)
        steps.append(rec)
        if is_stop:
            terminal = "stop"
            break
        meta = gated_metas[pool[a]]
        edits, kind = _edits_for(ast, meta)
        rec["edit_types"] = tuple(str(e.edit_type) for e in edits)
        rec["action_family"] = _proposal_detailed_family(ast, meta)
        rec["pair_action_cap"] = int(pair_cap) if action_space == "policy_sampled" else None
        _worker_heartbeat("execute_start", iid=iid, traj_id=traj_id, step=t,
                          action_family=rec["action_family"])
        t1 = time.time()
        replay_key = (iid, h, sig)
        if (action_space in ("validated", "multistep", "lexicographic") and
                replay_key in val_cache):
            # These action sets already ran the exact deterministic G1 replay
            # for every eligible proposal.  Reuse that successor for the chosen
            # action; the assertion below still verifies its recorded gain.
            res = val_cache[replay_key]
            execution_reason = "cached_validated"
        elif action_space == "policy_sampled":
            if anchor_successor is not None:
                res = anchor_successor
                execution_reason = "cached_anchor_probe"
            else:
                res, execution_reason, replay_hit = replay_cache.execute(
                    _execute_step_with_reason, executor, problem, schedule, edits, ms_cur, h)
                rec["execution_cache_hit"] = bool(replay_hit)
        else:
            res = _execute_step(executor, problem, schedule, edits, ms_cur, h)
            execution_reason = "ok" if res is not None else "infeasible"
        rec["execution_reason"] = execution_reason
        t_prof["execute"] += time.time() - t1
        _worker_heartbeat("execute_end", iid=iid, traj_id=traj_id, step=t,
                          execution_reason=str(execution_reason))
        u = None if res is None else float(res["improvement"])
        if action_space in ("validated", "multistep", "lexicographic") and res is not None:
            # R18 §52/53 (and its R19 §52-53 continuation): the executed gain
            # MUST equal the single-step replay G1 recorded at validation time
            # for this proposal (same replay semantics, deterministic).  Any
            # mismatch is a correctness bug -- STOP.
            _vrow = next(r for r in vs["pval"]["rows"] if r["k"] == pool[a])
            _exp = float(_vrow["g"] if action_space == "validated" else _vrow["g1"])
            if float(u) != _exp:
                raise AssertionError(
                    f"§52/53 validation/execute mismatch: "
                    f"validation_g={_exp!r} != executed_g={u!r} "
                    f"({iid}::{h} sig={sig}) -- correctness bug, STOP")
            rec["validation_gain"] = float(_vrow["g"] if action_space == "validated"
                                           else _vrow["g1"])
            rec["validation_tier"] = (_vrow["tier"] if action_space == "validated"
                                      else _vrow["class"])
            rec["validation_gain_norm"] = float(_vrow["gain_norm"] if action_space == "validated"
                                                else _vrow["gain_norm1"])
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
        rec["successor_makespan"] = int(res["schedule"].makespan)
        if getattr(C, 'E2E_LOAD_REWARD_ENABLED', False):
            from .load_reward import load_cost
            rec['load_cost_before'] = load_cost(problem, schedule)
            rec['load_cost_after'] = load_cost(problem, res['schedule'])
        if visual_plan is not None:
            selected_indices = ((meta["i"],) if meta["kind"] == "single"
                                else (meta["i"], meta["j"]))
            proposal_root_ops = sorted({
                str(root_op)
                for selected_index in selected_indices
                for root_op in (ast["pool"][selected_index].get("root_ops") or
                                (ast["pool"][selected_index]["e"].operation_id,))
            })
            appearance_rows = []
            appearance_payload = ast.get("appearance", {})
            raw_appearance_rows = (appearance_payload.get("blocks", ()) or
                                   appearance_payload.get("retained", ()))
            for appearance_item in raw_appearance_rows:
                if appearance_item.get("keep_or_prune", "keep") == "prune":
                    continue
                block = appearance_item.get("block", appearance_item)
                block_ops = tuple(str(op) for op in block.get("operations", ()))
                appearance_rows.append({
                    "block_id": str(block.get("block_id", "")),
                    "rules": tuple(str(rule) for rule in
                                   block.get("appearance_rules", ())),
                    "operations": block_ops,
                    "machines": tuple(str(machine) for machine in
                                      block.get("machines", ())),
                    "time_interval": tuple(block.get("time_interval", ())),
                    "appearance_values": dict(block.get("appearance_values", {})),
                    "priority": float(appearance_item.get("priority", 0.0)),
                    "directly_touches_selected_root": bool(
                        set(block_ops) & set(gate.get("retained", ()))),
                    "directly_touches_proposal_root": bool(
                        set(block_ops) & set(proposal_root_ops)),
                })
            visual_plan.append({
                "causal_paths": selected_root_paths(ast, gate.get("retained", ())),
                "step": int(step), "signature": str(sig), "kind": str(kind),
                "sampled_class": rec.get("sampled_class"),
                "active_layer": rec.get("active_layer"),
                "before_ms": int(ms_cur), "after_ms": int(res["schedule"].makespan),
                "improvement": float(res["improvement"]),
                "edits": tuple(copy.deepcopy(edits)),
                "m2_selected_roots": tuple(str(op) for op in
                                           gate.get("retained", ())),
                "proposal_root_ops": tuple(proposal_root_ops),
                "m2_selected_root_scores": {
                    str(op): float(ast.get("op_b5", {}).get(op, 0.0))
                    for op in gate.get("retained", ())
                },
                "m2_selected_root_hop": gate["diag"].get("selected_min_hop", -1),
                "m2_selected_root_support_count": gate["diag"].get(
                    "selected_support_count", 0),
                "m2_selected_root_support_appearances": tuple(
                    gate["diag"].get("selected_support_blocks", ())),
                "m2_selected_root_prior_trust": gate["diag"].get(
                    "selected_prior_trust", 1.0),
                "m2_selected_root_rl_value": gate["diag"].get(
                    "selected_root_value", 0.0),
                "m2_b5_top1_root": gate["diag"].get("b5_top1_root"),
                "m2_b5_top1_hop": gate["diag"].get("b5_top1_min_hop", -1),
                "reverse_message_passing_layers": int(
                    len(getattr(model_b5, "reverse_layers", ()))),
                "retained_appearance_blocks": tuple(appearance_rows),
            })
        n_acted += 1
        last_step_gain = float(res["improvement"])
        non_improving_streak = (0 if last_step_gain > 0.0
                                else non_improving_streak + 1)
        schedule = res["schedule"]
        ms_cur = int(schedule.makespan)
        nh = schedule_hash(schedule)
        pm.executed[iid][step]["successor_state_hash"] = nh
        if n_acted == max(1, horizon // 2):
            state_sample = {
                "schedule": copy.deepcopy(schedule), "ms": int(ms_cur),
                "n_acted": n_acted, "n_steps": t + 1,
                "records": tuple(copy.deepcopy(r) for s, r in
                    sorted(pm.executed.get(iid, {}).items())
                    if s not in root_exec_records or r != root_exec_records[s])}
        if nh in visited:
            terminal = "revisit"
            break
        visited.add(nh)
        if ms_cur < best_ms:
            best_ms = int(ms_cur)
            best_step = int(n_acted)
            best_schedule = copy.deepcopy(schedule)
            best_memory_records = [copy.deepcopy(rec) for s, rec in
                                   sorted(pm.executed.get(iid, {}).items())
                                   if s not in root_exec_records or
                                   rec != root_exec_records[s]]
        if stop_on_negative and float(res["improvement"]) <= 0.0:
            terminal = "non_positive"
            break
    # T2-L: one global net-intervention return for both M2 and M3.  Removing an
    # appearance is never rewarded directly: only the complete schedule's best
    # makespan counts, while moving away from that best schedule is the induced-
    # delay/downside term.  Thus congestion transferred to a new critical path
    # is automatically charged without enumerating new appearances.
    terminal_reward = int(root_ms) - ms_cur
    infeasible_penalty = (0.05 * float(root_ms) if terminal == "infeasible" else 0.0)
    best_reward = int(root_ms) - int(best_ms)
    best_to_terminal_regression = max(0.0, float(ms_cur - best_ms))
    reward = (float(best_reward)
              - float(C.T2L_NET_REGRESSION_WEIGHT) * best_to_terminal_regression
              - float(infeasible_penalty))
    # Every step receives the same semantics from its own state: improvement of
    # the best future schedule minus regression from that best to the terminal
    # schedule.  The current state is included in the minimum, so a rollout that
    # only gets worse has zero best gain and a strictly negative downside.
    _assign_t2l_future_credit(steps, terminal_ms=ms_cur, root_ms=root_ms)
    for local_step, rec in enumerate(steps, start=1):
        rec["reward_terminal"] = terminal_reward
        rec["reward_objective"] = reward
        rec["reward_best"] = float(best_reward)
        rec["reward_regression"] = best_to_terminal_regression
        rec["successful_anchor_prefix"] = bool(
            rec.get("is_anchor", False) and reward > 0 and
            local_step <= int(best_step))
    # In score-sampled T2-G, M2 is part of the same hierarchical action and is
    # trained from the same terminal group advantage as M3.  There is no local
    # probe-derived q2 because no candidate is executed before policy sampling.
    U2 = (float(reward) if action_space == "policy_sampled" else
          (float(np.mean(step_u2)) if step_u2 else 0.0))
    assert h0 == schedule_hash(schedule_root), "sibling State isolation violated"
    terminal_records = [copy.deepcopy(rec) for s, rec in
                        sorted(pm.executed.get(iid, {}).items())
                        if s not in root_exec_records or
                        rec != root_exec_records[s]]
    return {
        "traj_id": traj_id, "iid": iid, "episode_id": episode_id,
        "root_state_hash": h0, "root_ms": int(root_ms), "final_ms": ms_cur,
        "state_sample": state_sample,
        "reward": reward, "terminal_reward": terminal_reward,
        "infeasible_penalty": float(infeasible_penalty),
        "net_intervention_reward": float(reward),
        "best_to_terminal_regression": best_to_terminal_regression,
        "net_regression_weight": float(C.T2L_NET_REGRESSION_WEIGHT),
        # Success means the trajectory improved the incumbent at least once;
        # the terminal schedule may regress afterwards and is only diagnostic.
        "best_reward": best_reward, "success": int(best_reward > 0),
        "best_ms": int(best_ms), "best_step": int(best_step),
        "is_anchor": bool(action_space == "policy_sampled" and
                          isinstance(traj_id, int) and
                          0 <= traj_id < int(anchor_trajectories)),
        "reward_semantics": "unified_net_intervention_return",
        "U2": U2, "m2_reward_breakdown": m2_breakdown,
        "m2_credit_source": ("net_best_within_remaining_horizon"
                             if action_space == "policy_sampled" else "probe_q2"),
        "initial_selected_root": (steps[0].get("m2_diag", {}).get("selected_root")
                                  if steps else None),
        "initial_selected_roots": (tuple(steps[0].get("m2_diag", {}).get(
                                      "selected_roots", ())) if steps else ()),
        "initial_root_set_key": canonical_root_set(
            steps[0].get("m2_diag", {}).get("selected_roots", ())) if steps else (),
        "replay_cache_lookups": replay_cache.lookups - replay_start[0],
        "replay_cache_hits": replay_cache.hits - replay_start[1],
        "real_execution_calls": replay_cache.executions - replay_start[2],
        "root_group_id": root_group_id,
        "n_steps": len(steps), "n_acted": n_acted,
        "terminal": terminal, "steps": steps,
        "mem_probe": mem_probe, "n_mem_records": _mem_record_count(pm),
        "root_mem_records": _mem_record_count(progmem_root),
        "memory_lineage": {"root_records": _mem_record_count(progmem_root),
                           "branch_records": _mem_record_count(pm),
                           "probe": mem_probe},
        "prof": t_prof, "semantics": "true_stop", "joint": True, "stagewise": True,
        "vprobe_n": vprobe_n, "vprobe_ms": vprobe_ms,
        "post_m2_root_gate_bypass_attempts": post_m2_root_gate_bypass_attempts,
        "pval_nonfinite_g1_count": pval_nonfinite_g1_count,
        "pval_nonfinite_gh_count": pval_nonfinite_gh_count,
        "preselection_replay_count": (feasibility_replay_count
                                      if action_space == "policy_sampled" else None),
        "selected_action_replay_count": sum(
            1 for rec in steps if not bool(rec.get("is_stop", False))),
        # Keep both terminal and best-prefix states: best-within-horizon drives
        # GRPO and frontier promotion; terminal regression is diagnostic only.
        "terminal_schedule": copy.deepcopy(schedule),
        "terminal_state_hash": schedule_hash(schedule),
        "terminal_memory_records": terminal_records,
        "best_schedule": best_schedule,
        "best_state_hash": schedule_hash(best_schedule),
        "best_memory_records": best_memory_records,
        "_visual_plan": visual_plan,
    }


def _assign_t2l_future_credit(steps, *, terminal_ms, root_ms):
    """Attach exact per-state best-within-remaining-horizon M2 credit.

    This helper is deliberately independent of the group advantage.  Each M2
    decision is judged from the schedule that existed immediately before that
    decision, not from the common S0/root makespan.
    """
    future_min_ms = int(terminal_ms)
    for rec in reversed(steps):
        successor_ms = rec.get("successor_makespan")
        if successor_ms is not None:
            future_min_ms = min(future_min_ms, int(successor_ms))
        before_ms = int(rec.get("state_makespan_before", root_ms))
        reference_best_ms = min(before_ms, future_min_ms)
        local_penalty = (0.05 * float(before_ms)
                         if rec.get("execution_reason") == "infeasible" else 0.0)
        future_best_gain = float(before_ms - reference_best_ms)
        future_regression = max(0.0, float(terminal_ms - reference_best_ms))
        future_net_reward = (
            future_best_gain
            - float(C.T2L_NET_REGRESSION_WEIGHT) * future_regression
            - local_penalty)
        rec["m2_future_best_ms"] = int(reference_best_ms)
        rec["m2_future_best_reward"] = future_best_gain
        rec["m2_future_terminal_reward"] = (
            float(before_ms - int(terminal_ms)) - local_penalty)
        rec["m2_future_regression"] = future_regression
        rec["m2_future_net_reward"] = future_net_reward
        rec["m2_future_best_success"] = int(
            future_best_gain > 0.0)
    return steps


class _BoundedWorkerReplayCache(dict):
    """Hard-cap a worker-local pure memo without changing replay results."""

    def __init__(self, initial=None, limit=None):
        super().__init__()
        self.limit = max(1, int(
            getattr(C, "T2L_WORKER_REPLAY_CACHE_ENTRIES", 1000)
            if limit is None else limit))
        for key, value in (initial or {}).items():
            self[key] = value

    def __setitem__(self, key, value):
        if key not in self and len(self) >= self.limit:
            # Values are deterministic pure replay results, so eviction affects
            # only future recomputation time.  Clearing also drops references to
            # large schedule/proposal objects before the process grows forever.
            self.clear()
        super().__setitem__(key, value)


def _mp_worker_init_r14(jpol, scorer, executor, model_b5, single_head, direct_head,
                        val_cache=None, critical_gate=False, torch_threads=1):
    load_upstream()
    # These workers run many tiny CPU forwards.  Letting every process inherit
    # the host-wide BLAS/OpenMP thread count creates severe oversubscription
    # (e.g. 16 processes x 25 threads).  One intra-op thread per rollout worker
    # is the throughput-safe default and does not change model parameters or the
    # rollout/update schedule.
    try:
        torch.set_num_threads(max(1, int(torch_threads)))
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if critical_gate:
        _install_worker_critical_gate()
    global _MP14_STATE
    _MP14_STATE = {
        "jpol": jpol, "scorer": scorer, "executor": executor,
        "model_b5": model_b5, "single_head": single_head, "direct_head": direct_head,
        "val_cache": _BoundedWorkerReplayCache(
            val_cache, limit=getattr(C, "T2L_WORKER_REPLAY_CACHE_ENTRIES", 1000)),
    }


def _install_worker_critical_gate():
    """Install the T2-C critical-path pool gate inside a spawned worker.

    A parent-process monkeypatch is not inherited by multiprocessing ``spawn``.
    Keeping this small worker-local wrapper makes collection match the runner's
    gated evaluation without relying on fork-only process state.
    """
    from . import proposal_features as pf
    from .critical_path import critical_path_info

    orig = pf.analyze_state
    if getattr(orig, "_t2c_worker_critical_gate", False):
        return
    cache = {}

    def gated(problem, schedule, model_b5, case_id):
        out = orig(problem, schedule, model_b5, case_id)
        try:
            key = (schedule.problem_id, schedule_hash(schedule))
            info = cache.get(key)
            if info is None:
                info = critical_path_info(problem, schedule)
                cache[key] = info
            pool = list(out.get("pool") or [])
            crit_ops = info["critical_ops"]
            kept = [r for r in pool if r.get("e") is not None
                    and r["e"].operation_id in crit_ops]
            if kept:
                out["pool"] = kept
            out["critical_info"] = {
                "makespan": info["makespan"], "n_ops": info["n_ops"],
                "n_critical": info["n_critical"],
                "critical_ops": sorted(crit_ops),
            }
            out["gated_original_pool"] = len(pool)
        except Exception:  # gate failure preserves the canonical full pool
            pass
        return out

    gated._t2c_worker_critical_gate = True
    pf.analyze_state = gated


def _finish_group_r14(trajs, iid, h0, root_ms, progmem, coll_s, workers,
                      parallelism="mp", batch_coll_s=None):
    """Attach group-relative credit and build the canonical R14 group bundle."""
    mem_unchanged = all(tr["mem_probe"] is not None and
                        tr["root_mem_records"] == _mem_record_count(progmem)
                        for tr in trajs)
    policy_sampled_semantics = {
        "terminal_primary+best_prefix_aux+success_aux",  # old T2-H/T2-I resume
        "m2_best_primary+m3_terminal_primary+future_best_rtg",
        "m2+m3_best_primary+future_best_rtg",
        "unified_net_intervention_return",
    }
    policy_sampled = any(tr.get("reward_semantics") in policy_sampled_semantics
                         for tr in trajs)
    pure_indices = [i for i, tr in enumerate(trajs)
                    if not (policy_sampled and tr.get("is_anchor", False))]

    def _pure_advantages(values):
        """GRPO normalization over on-policy siblings only; anchors stay auxiliary."""
        indices = pure_indices or list(range(len(values)))
        sub = group_advantages_r12([float(values[i]) for i in indices])
        expanded = [0.0] * len(values)
        for i, value in zip(indices, sub["advantages"]):
            expanded[i] = float(value)
        return {**sub, "advantages": expanded}

    def _hierarchical_advantages(values):
        """M3 compares trajectories inside one root; M2 compares root returns."""
        indices = pure_indices or list(range(len(values)))
        if not any(trajs[i].get("initial_selected_root") is not None for i in indices):
            legacy = _pure_advantages(values)
            return {"inner": legacy["advantages"],
                    "outer": legacy["advantages"],
                    "inner_informative": bool(legacy["informative"]),
                    "outer_informative": bool(legacy["informative"]),
                    "unique_roots": 0, "root_returns": [], "root_keys": [],
                    "outer_stats": legacy}
        multi_root = any(len(tuple(trajs[i].get("initial_selected_roots") or ())) > 1
                         for i in indices)
        by_root = {}
        for i in indices:
            if multi_root:
                selected = tuple(trajs[i].get("initial_selected_roots") or ())
                key = ("ROOTSET::" + ">".join(map(str, selected))
                       if selected else f"missing::{i}")
            else:
                root = trajs[i].get("initial_selected_root")
                key = str(root) if root is not None else f"missing::{i}"
            by_root.setdefault(key, []).append(i)
        if multi_root:
            # One state -> one sampled root SET -> one unioned M3 action pool.
            # M3 compares proposal outcomes across the whole same-state sibling
            # group.  M2 compares complete root-set returns, never four roots as
            # four independent actions.  Repeated identical sets are averaged.
            global_inner = _pure_advantages(values)
            inner = list(global_inner["advantages"])
            inner_info = bool(global_inner["informative"])
        else:
            inner = [0.0] * len(values)
            inner_info = False
            for members in by_root.values():
                sub = group_advantages_r12([float(values[i]) for i in members])
                inner_info = inner_info or bool(sub["informative"])
                for i, advantage in zip(members, sub["advantages"]):
                    inner[i] = float(advantage)
        root_keys = list(by_root)
        root_returns = [float(np.mean([values[i] for i in by_root[key]]))
                        for key in root_keys]
        outer_sub = group_advantages_r12(root_returns)
        outer = [0.0] * len(values)
        for key, advantage in zip(root_keys, outer_sub["advantages"]):
            for i in by_root[key]:
                outer[i] = float(advantage)
        return {"inner": inner, "outer": outer,
                "multi_root_joint_inner": bool(multi_root),
                "multi_root_joint_outer": bool(multi_root),
                "inner_informative": bool(inner_info),
                "outer_informative": bool(outer_sub["informative"]),
                "unique_roots": len(by_root), "root_returns": root_returns,
                "root_keys": root_keys, "outer_stats": outer_sub}

    def _stepwise_advantages(field):
        """GRPO baseline for diverged states at the same remaining horizon.

        After step 1, siblings no longer share an identical schedule, so a
        controlled root/proposal-within-state comparison is impossible without
        another exponential rollout tree.  We still train every on-policy M2
        and M3 decision: compare its future-best return-to-go with peer decisions
        at the same local timestep from the same original root group.
        """
        out, informative = {}, {}
        indices = pure_indices or list(range(len(trajs)))
        max_steps = max((len(trajs[i].get("steps", ())) for i in indices), default=0)
        for local_index in range(max_steps):
            members = [(i, trajs[i]["steps"][local_index]) for i in indices
                       if local_index < len(trajs[i].get("steps", ())) and
                       field in trajs[i]["steps"][local_index]]
            if not members:
                continue
            sub = group_advantages_r12([float(rec[field]) for _, rec in members])
            informative[local_index] = bool(sub["informative"])
            for (i, _rec), advantage in zip(members, sub["advantages"]):
                out[(i, local_index)] = float(advantage)
        return out, informative

    rewards3 = [float(tr["reward"]) for tr in trajs]
    U2v = [float(tr.get("U2", 0.0)) for tr in trajs]
    summary_adv = _pure_advantages(rewards3)
    if policy_sampled:
        # L normalizes exactly one scalar target.  This avoids K's mathematically
        # different result from separately standardizing best, terminal and
        # success before adding their advantages.
        h_net = _hierarchical_advantages(rewards3)
        h_terminal = h_net  # compatibility name used by group diagnostics
        a3_rows, a2_rows = h_net["inner"], h_net["outer"]
        ab3_rows, ab2_rows = a3_rows, a2_rows
        az3_rows = az2_rows = [0.0] * len(trajs)
        step_net, step_net_info = _stepwise_advantages("m2_future_net_reward")
        later_info = bool(any(step_net_info.values()))
        m3_info = bool(h_net["inner_informative"] or later_info)
        m2_info = bool(h_net["outer_informative"] or later_info)
    else:
        adv2 = _pure_advantages(U2v)
        adv_best = _pure_advantages(
            [float(tr.get("best_reward", tr["reward"])) for tr in trajs])
        adv_success = _pure_advantages(
            [float(tr.get("success", float(tr["reward"]) > 0.0)) for tr in trajs])
        a3_rows, a2_rows = summary_adv["advantages"], adv2["advantages"]
        ab3_rows = ab2_rows = adv_best["advantages"]
        az3_rows = az2_rows = adv_success["advantages"]
        m3_info = bool(summary_adv["informative"] or adv_best["informative"] or
                       adv_success["informative"])
        m2_info = bool(adv2["informative"])
        h_terminal = {"unique_roots": 0, "root_returns": [], "root_keys": []}
    for tr_index, (tr, a3, a2, ab3, ab2, az3, az2) in enumerate(zip(
            trajs, a3_rows, a2_rows, ab3_rows, ab2_rows, az3_rows, az2_rows)):
        prefix_end = int(tr.get("best_step", len(tr["steps"])))
        for local_step, rec in enumerate(tr["steps"], start=1):
            if policy_sampled and tr.get("is_anchor", False):
                imitate = bool(rec.get("successful_anchor_prefix", False))
                rec["adv3_terminal"] = 0.0
                rec["adv3_best_prefix"] = 0.0
                rec["adv3_success"] = 0.05 if imitate else 0.0
                rec["adv3"] = 0.05 if imitate else 0.0
                rec["inf3"] = imitate
                rec["adv2"] = 0.05 if imitate else 0.0
                rec["inf2"] = imitate
                rec["credit_source"] = "successful_anchor_prefix_imitation"
                rec["grp_key"] = (iid, h0)
                continue
            prefix_credit3 = prefix_credit2 = 0.0
            success_credit3 = success_credit2 = 0.0
            # M2 and M3 share one net-makespan return.  At the common first state
            # M3 compares proposals within a root and M2 compares root returns;
            # later decisions use the same-timestep net return-to-go.
            if policy_sampled and local_step == 1:
                combined3 = float(a3)
                local_m3_info = bool(h_net["inner_informative"])
                m3_source = "M3_inner_proposal_unified_net_return"
                m3_best_component = float(a3)
                m3_terminal_component = 0.0
                m3_success_component = 0.0
            elif policy_sampled:
                key = (tr_index, local_step - 1)
                combined3 = float(step_net.get(key, 0.0))
                m3_best_component = combined3
                m3_terminal_component = 0.0
                m3_success_component = 0.0
                local_m3_info = bool(step_net_info.get(local_step - 1, False))
                m3_source = "M3_stepwise_unified_net_return_to_go"
            else:
                combined3 = float(a3) + float(prefix_credit3) + float(success_credit3)
                local_m3_info = m3_info
                m3_source = "canonical_stagewise"
                m3_best_component = float(prefix_credit3)
                m3_terminal_component = float(a3)
                m3_success_component = float(success_credit3)
            if policy_sampled and local_step == 1:
                combined2 = float(a2)
                local_m2_info = m2_info
                m2_source = "M2_outer_rootset_unified_net_return"
            elif policy_sampled:
                key = (tr_index, local_step - 1)
                combined2 = float(step_net.get(key, 0.0))
                local_m2_info = bool(step_net_info.get(local_step - 1, False))
                m2_source = "M2_stepwise_unified_net_return_to_go"
            else:
                combined2 = float(a2) + float(prefix_credit2) + float(success_credit2)
                local_m2_info = m2_info
                m2_source = "canonical_stagewise"
            rec["adv3_terminal"] = m3_terminal_component
            rec["adv3_best_prefix"] = m3_best_component
            rec["adv3_success"] = m3_success_component
            rec["adv3"] = combined3
            rec["inf3"] = bool(local_m3_info)
            rec["adv2"] = combined2
            rec["inf2"] = bool(local_m2_info)
            rec["m2_credit_source"] = m2_source
            rec["m3_credit_source"] = m3_source
            rec["credit_source"] = (
                "hierarchical_grpo:M2+M3_unified_net_intervention"
                if policy_sampled else "canonical_stagewise")
            rec["grp_key"] = (iid, h0)
    prof_keys = ("analyze", "proposal_structure", "m2gate", "mem",
                 "m3_features", "policy", "execute", "validation")
    prof_cpu_s = {key: sum(float(tr.get("prof", {}).get(key, 0.0)) for tr in trajs)
                  for key in prof_keys}
    cache_lookups = sum(int((rec.get("pv_diag") or {}).get("cache_lookups", 0))
                        for tr in trajs for rec in tr.get("steps", []))
    cache_hits = sum(int((rec.get("pv_diag") or {}).get("cache_hits", 0))
                     for tr in trajs for rec in tr.get("steps", []))
    anchor_informative = any(
        bool(tr.get("is_anchor", False)) and
        float(tr.get("net_intervention_reward", tr.get("reward", 0.0))) > 0.0
        for tr in trajs)
    def _path_signature(tr):
        return tuple(str(rec.get("proposal_signature") or
                         rec.get("action_signature") or "")
                     for rec in tr.get("steps", ()))
    all_signatures = [_path_signature(tr) for tr in trajs]
    pure_signatures = [_path_signature(tr) for tr in trajs
                       if not tr.get("is_anchor", False)]
    # Real rollouts always carry terminal_schedule.  Keep the group finisher
    # tolerant of legacy/test bundles so diagnostics never break training.
    terminal_hashes = [
        schedule_hash(tr["terminal_schedule"])
        if tr.get("terminal_schedule") is not None
        else ("missing_terminal_schedule", _path_signature(tr), tr.get("terminal"))
        for tr in trajs
    ]
    out = {
        "grp_key": (iid, h0), "iid": iid, "state_hash": h0,
        "root_ms": int(root_ms),
        "rewards": [float(x) for x in np.asarray(
            [tr["reward"] for tr in trajs], dtype=np.float64).tolist()],
        "terminal_rewards": [float(tr.get("terminal_reward", tr["reward"]))
                             for tr in trajs],
        "net_intervention_rewards": [float(
            tr.get("net_intervention_reward", tr["reward"])) for tr in trajs],
        "best_to_terminal_regressions": [float(
            tr.get("best_to_terminal_regression", 0.0)) for tr in trajs],
        "mean_reward": summary_adv["mean"], "std_reward": summary_adv["std"],
        "informative": bool(m3_info or anchor_informative),
        "info2": bool(m2_info or anchor_informative),
        "advantages": a3_rows, "U2": U2v,
        "U2_mean": float(np.mean(U2v)) if U2v else 0.0,
        "U2_std": float(np.std(U2v)) if U2v else 0.0,
        "advantages2": a2_rows, "trajs": trajs,
        "hierarchical_credit": bool(policy_sampled),
        "unique_initial_roots": int(h_terminal.get("unique_roots", 0)),
        "root_returns": dict(zip(h_terminal.get("root_keys", ()),
                                 h_terminal.get("root_returns", ()))),
        "best_rewards": [float(tr.get("best_reward", tr["reward"]))
                         for tr in trajs],
        "successes": [int(tr.get("success", float(tr["reward"]) > 0.0))
                      for tr in trajs],
        "m2_credit_weights": {"unified_net_intervention": 1.0,
                              "regression_weight": float(
                                  C.T2L_NET_REGRESSION_WEIGHT)},
        "m3_credit_weights": {"unified_net_intervention": 1.0,
                              "regression_weight": float(
                                  C.T2L_NET_REGRESSION_WEIGHT)},
        "anchor_trajectories": sum(bool(tr.get("is_anchor", False)) for tr in trajs),
        "anchor_successful_prefixes": sum(
            bool(tr.get("is_anchor", False)) and float(tr.get("best_reward", 0.0)) > 0.0
            for tr in trajs),
        "unique_trajectory_count": len(set(all_signatures)),
        "unique_trajectory_rate": len(set(all_signatures)) / max(len(all_signatures), 1),
        "diversity_v4": sibling_diversity(trajs),
        "pure_unique_trajectory_count": len(set(pure_signatures)),
        "pure_unique_trajectory_rate": (
            len(set(pure_signatures)) / max(len(pure_signatures), 1)),
        "unique_terminal_state_count": len(set(terminal_hashes)),
        "unique_terminal_state_rate": len(set(terminal_hashes)) / max(len(terminal_hashes), 1),
        "n_steps": sum(tr["n_steps"] for tr in trajs),
        "n_acted": sum(tr["n_acted"] for tr in trajs),
        "terminal_counts": dict(Counter(tr["terminal"] for tr in trajs)),
        "coll_s": float(coll_s), "mem_unchanged": mem_unchanged,
        "parallelism": parallelism, "workers": int(workers),
        "stagewise": True,
        "vprobe_n": sum(tr.get("vprobe_n", 0) for tr in trajs),
        "vprobe_ms": sum(tr.get("vprobe_ms", 0.0) for tr in trajs),
        "prof_cpu_s": prof_cpu_s,
        "validation_cache_lookups": cache_lookups,
        "validation_cache_hits": cache_hits,
        "validation_cache_hit_rate": (cache_hits / max(cache_lookups, 1)),
    }
    if batch_coll_s is not None:
        out["batch_coll_s"] = float(batch_coll_s)
    return out


def _initial_root_stratum(seed, iid, episode_id, state_hash, kid, k,
                          anchor_trajectories, per_root=None):
    """Randomly rotated first-root strata; v4 defaults to one sibling per stratum.

    A uniformly random *rotation* of an evenly spaced grid makes every fixed
    stratum marginally distributed as the M2 categorical policy, while siblings
    in one stratum deliberately reuse the same root so M3 can be compared
    conditionally without extra trajectories.  Do not divide the offset by the
    number of groups: `(group+offset)/G` would confine each stratum to one CDF
    interval and make its stored categorical old log-prob incorrect.
    """
    per_root = (int(C.T2L_M2_SIBLINGS_PER_ROOT) if per_root is None else int(per_root))
    pure = max(int(k) - int(anchor_trajectories), 0)
    if int(kid) < int(anchor_trajectories) or pure <= 0:
        return None, None
    pure_id = int(kid) - int(anchor_trajectories)
    group_id = pure_id // max(int(per_root), 1)
    n_groups = max(1, int(math.ceil(pure / max(int(per_root), 1))))
    common_seed = _traj_seed(seed, iid, episode_id, state_hash, 99173)
    offset = random.Random(common_seed).random()
    return (offset + group_id / n_groups) % 1.0, int(group_id)


def _initial_action_stratum(seed, iid, episode_id, state_hash, kid, k,
                            anchor_trajectories):
    pure = int(k) - int(anchor_trajectories)
    index = int(kid) - int(anchor_trajectories)
    if not bool(C.T2L_M3_FIRST_STEP_STRATIFIED) or not 0 <= index < pure:
        return None
    # Separate namespace: never reuse the M2 rotation/pool-generation RNG.
    return (_traj_seed(seed, iid, episode_id, state_hash, 190771), index, pure)


def _new_replay_cache():
    return BoundedReplayCache(C.T2L_ROLLOUT_REPLAY_CACHE_ENTRIES,
                             int(C.T2L_ROLLOUT_REPLAY_CACHE_MB * 1024 * 1024))


def _new_sibling_analysis_cache(model_b5, single_head, direct_head):
    return (AnalyzeCache(model_b5, single_head, direct_head)
            if bool(C.T2L_SHARE_SIBLING_ANALYZE_CACHE) else None)


def _mp_traj_job_r14(contract):
    st = _MP14_STATE
    return collect_trajectory_r14(
        st["jpol"], st["scorer"], st["executor"],
        st["model_b5"], st["single_head"], st["direct_head"],
        contract["problem"], contract["schedule_root"], contract["root_ms"],
        contract["iid"], contract["episode_id"], contract["progmem_root"],
        seed=contract["seed"], traj_id=contract["traj_id"],
        T=contract.get("T"), eps=contract.get("eps"), horizon=contract.get("horizon"),
        step_offset=contract.get("step_offset", 0), seed_rolex=contract.get("seed_rolex"),
        seed_ast=contract.get("seed_ast"),
        action_space=contract.get("action_space", "full"),
        val_cache=st["val_cache"],
        stop_on_negative=contract.get("stop_on_negative", True),
        allow_policy_stop=contract.get("allow_policy_stop", True),
        feasible_fallback=contract.get("feasible_fallback", False),
        capture_visual_plan=contract.get("capture_visual_plan", False),
        anchor_trajectories=contract.get("anchor_trajectories", 0),
        initial_root_quantile=contract.get("initial_root_quantile"),
        root_group_id=contract.get("root_group_id"),
        initial_action_stratum=contract.get("initial_action_stratum"))


def collect_full_group_rollouts_r14(jpol, scorer, executor, model_b5, single_head,
                                    direct_head, problem, schedule, root_ms, iid,
                                    episode_id, progmem, k=None, T=None, eps=None,
                                    horizon=None, seed=0, step_offset=0, workers=1,
                                    step0_cache=None, mp_ctx=None, action_space="full",
                                    val_cache=None, stop_on_negative=True,
                                    allow_policy_stop=True, feasible_fallback=False,
                                    capture_visual_plans=False):
    """K sibling trajectories under R14 with TWO independent group advantages:
    A3 from terminal rewards (§16-17, M3) and A2 from per-trajectory U2 (§13-14, M2),
    both over the SAME K sibling set at one root state.  Probe gains never enter A3;
    terminal makespan never enters A2 -- the §15 hard-credit separation.
    workers=1 == workers=N bit-identical (same deterministic collect)."""
    from concurrent.futures import ProcessPoolExecutor
    k = int(k if k is not None else C.TO1_R13_K)
    anchor_trajectories = (
        min(max(0, int(getattr(C, "T2L_ANCHOR_TRAJECTORIES", 2))),
            max(k - 2, 0)) if action_space == "policy_sampled" else 0)
    h0 = schedule_hash(schedule)
    seed_rolex = seed_ast = None
    seed_ast_s = 0.0
    if step0_cache is not None:
        seed_rolex, seed_ast, seed_ast_s = _seed_state13(
            step0_cache, problem, schedule, iid)

    def _contract(kid):
        root_q, root_gid = _initial_root_stratum(
            seed, iid, episode_id, h0, kid, k, anchor_trajectories)
        return {
            "seed": _traj_seed(seed, iid, episode_id, h0, kid), "traj_id": kid,
            "problem": problem, "schedule_root": schedule, "root_ms": int(root_ms),
            "iid": iid, "episode_id": episode_id, "progmem_root": progmem,
            "T": T, "eps": eps, "horizon": horizon, "step_offset": step_offset,
            "seed_rolex": seed_rolex, "action_space": action_space,
            "stop_on_negative": bool(stop_on_negative),
            "allow_policy_stop": bool(allow_policy_stop),
            "feasible_fallback": bool(feasible_fallback),
            "capture_visual_plan": bool(capture_visual_plans),
            "anchor_trajectories": anchor_trajectories,
            "initial_root_quantile": root_q, "root_group_id": root_gid,
            "initial_action_stratum": _initial_action_stratum(
                seed, iid, episode_id, h0, kid, k, anchor_trajectories),
            "seed_ast": seed_ast, "seed_ast_s": seed_ast_s,
            "val_cache": ({} if val_cache is None else val_cache),
        }

    shared_analysis = _new_sibling_analysis_cache(model_b5, single_head, direct_head)
    shared_replay = _new_replay_cache()
    if seed_ast is None and shared_analysis is not None:
        seed_rolex, seed_ast, seed_ast_s = _seed_state13(shared_analysis, problem, schedule, iid)
    t0 = time.time()
    trajs = []
    if workers and int(workers) > 1:
        ctx = _resolve_mp_ctx(mp_ctx)
        with ProcessPoolExecutor(
                max_workers=int(workers), mp_context=ctx,
                initializer=_mp_worker_init_r14,
                initargs=(jpol, scorer, executor, model_b5, single_head, direct_head,
                          val_cache)) as ex:
            futs = [ex.submit(_mp_traj_job_r14, _contract(kid)) for kid in range(k)]
            for f in futs:
                trajs.append(f.result())
    else:
        for kid in range(k):
            c = _contract(kid)
            trajs.append(collect_trajectory_r14(
                jpol, scorer, executor, model_b5, single_head, direct_head,
                c["problem"], c["schedule_root"], c["root_ms"], c["iid"], c["episode_id"],
                c["progmem_root"], seed=c["seed"], traj_id=c["traj_id"],
                T=T, eps=eps, horizon=horizon, step_offset=step_offset,
                seed_rolex=seed_rolex, seed_ast=seed_ast, action_space=action_space,
                val_cache=({} if val_cache is None else val_cache),
                stop_on_negative=bool(stop_on_negative),
                allow_policy_stop=bool(allow_policy_stop),
                feasible_fallback=bool(feasible_fallback),
                capture_visual_plan=bool(capture_visual_plans),
                anchor_trajectories=anchor_trajectories,
                initial_root_quantile=c["initial_root_quantile"],
                root_group_id=c["root_group_id"],
                initial_action_stratum=c["initial_action_stratum"],
                analyze_cache=shared_analysis, replay_cache=shared_replay))
    coll_s = time.time() - t0

    out = _finish_group_r14(
        trajs, iid, h0, root_ms, progmem, coll_s, workers,
        parallelism=("mp" if (workers and int(workers) > 1) else "serial"))
    out["ast_profile"] = {"ast_calls_before": int(k), "ast_calls_after": 1,
                          "ast_seconds_per_call": float(seed_ast_s),
                          "ast_seconds_saved": float(seed_ast_s * max(k - 1, 0)),
                          "breakdown": dict((seed_ast or {}).get("_profile", {}))}
    return out


_MP14_BATCH_ROOTS = None


def _sibling_shard_jobs(n_graphs, k, workers, max_shards_per_graph=4):
    """Deterministically cover every (graph, sibling) exactly once.

    Graph-level tasks retain cache locality while there are enough active graphs.
    Once rolling termination leaves fewer graphs than workers, split each K-sibling
    group into at most ``max_shards_per_graph`` tasks.  The cap avoids multiplying
    the deterministic root validation across all workers for a single straggler.
    """
    n_graphs, k, workers = int(n_graphs), int(k), int(workers)
    if n_graphs < 1 or k < 1:
        return []
    cap = max(1, int(max_shards_per_graph))
    # Preserve graph locality while there are enough independent roots to occupy
    # the pool.  The previous implementation always split K=24 into four shards,
    # so 24 roots became ~96 queued jobs even with only 8-24 workers; that
    # multiplied root serialization, cache state and IPC with no parallelism gain.
    # Only shard siblings once active graph count drops below worker count.
    if n_graphs >= workers:
        shards_per_graph = 1
    else:
        target_siblings = 6
        load_balance_shards = max(1, int(math.ceil(k / target_siblings)))
        needed_for_workers = max(1, int(math.ceil(workers / n_graphs)))
        shards_per_graph = min(k, cap, load_balance_shards, needed_for_workers)
    jobs = []
    for gi in range(n_graphs):
        for shard in range(shards_per_graph):
            kids = tuple(range(shard, k, shards_per_graph))
            if kids:
                jobs.append((gi, kids))
    return jobs


def _mp_worker_init_r14_batch(jpol, scorer, executor, model_b5, single_head,
                              direct_head, roots, val_cache=None,
                              critical_gate=False, torch_threads=1):
    _mp_worker_init_r14(
        jpol, scorer, executor, model_b5, single_head, direct_head,
        val_cache=val_cache, critical_gate=critical_gate,
        torch_threads=torch_threads)
    global _MP14_BATCH_ROOTS
    _MP14_BATCH_ROOTS = roots


def _mp_traj_job_r14_batch(job):
    group_idx, kids = job
    root = _MP14_BATCH_ROOTS[group_idx]
    st = _MP14_STATE
    shared_analysis = _new_sibling_analysis_cache(
        st["model_b5"], st["single_head"], st["direct_head"])
    shared_replay = _new_replay_cache()
    out = []
    for kid in kids:
        root_q, root_gid = _initial_root_stratum(
            root["seed"], root["iid"], root["episode_id"], root["state_hash"],
            kid, root.get("branches", C.TO1_R13_K),
            root.get("anchor_trajectories", 0))
        tr = collect_trajectory_r14(
            st["jpol"], st["scorer"], st["executor"], st["model_b5"],
            st["single_head"], st["direct_head"], root["problem"],
            root["schedule"], root["root_ms"], root["iid"], root["episode_id"],
            root["progmem"],
            seed=_traj_seed(root["seed"], root["iid"], root["episode_id"],
                            root["state_hash"], kid),
            traj_id=kid, T=root["T"], eps=root["eps"], horizon=root["horizon"],
            step_offset=root["step_offset"], seed_rolex=root["seed_rolex"],
            seed_ast=root.get("seed_ast"),
            action_space=root["action_space"], val_cache=st["val_cache"],
            analyze_cache=shared_analysis, replay_cache=shared_replay,
            stop_on_negative=root.get("stop_on_negative", True),
            allow_policy_stop=root.get("allow_policy_stop", True),
            feasible_fallback=root.get("feasible_fallback", False),
            capture_visual_plan=root.get("capture_visual_plans", False),
            anchor_trajectories=root.get("anchor_trajectories", 0),
            initial_root_quantile=root_q, root_group_id=root_gid,
            initial_action_stratum=_initial_action_stratum(
                root["seed"], root["iid"], root["episode_id"], root["state_hash"],
                kid, root.get("branches", C.TO1_R13_K), root.get("anchor_trajectories", 0)))
        out.append((kid, tr))
    return group_idx, out


def make_graph_rollout_pool(jpol, scorer, executor, model_b5, single_head,
                            direct_head, workers, mp_ctx=None, val_cache=None,
                            critical_gate=False, torch_threads=1):
    """Create the one long-lived graph-level rollout pool owned by the runner.

    Workers keep frozen backbones, scorer and replay machinery resident.  The
    small trainable actor snapshot is refreshed once per submitted depth, before
    that graph's K siblings are collected sequentially in the same worker.
    """
    from concurrent.futures import ProcessPoolExecutor
    ctx = _resolve_mp_ctx(mp_ctx)
    rollout_jpol = copy.deepcopy(jpol)
    # Adam moments and resume payloads belong only to the parent/GPU updater.
    # Removing them from the worker template avoids one large copy per process.
    for attr in ("_t2d_optimizer", "_t2d_optimizer_param_ids",
                 "_t2d_resume_optimizer_state"):
        rollout_jpol.__dict__.pop(attr, None)
    return ProcessPoolExecutor(
        max_workers=max(1, int(workers)), mp_context=ctx,
        initializer=_mp_worker_init_r14,
        initargs=(rollout_jpol, scorer, executor, model_b5, single_head, direct_head,
                  val_cache, bool(critical_gate), int(torch_threads)))


def _mp_graph_job_r14(job):
    """Collect one complete graph group under exactly one pi_old snapshot."""
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    group_idx, policy_snapshot, root, k = job
    _worker_heartbeat("job_start", iid=root["iid"],
                      siblings=int(k), group_idx=int(group_idx))
    st = _MP14_STATE
    # Validation replay memoization is a pure speed cache.  Persistent workers
    # otherwise retain every state/proposal object for the entire multi-day run
    # and can be SIGKILLed by the cgroup OOM killer.
    if len(st["val_cache"]) > int(getattr(
            C, "T2L_WORKER_REPLAY_CACHE_ENTRIES", 1000)):
        st["val_cache"].clear()
    cache_before = len(st["val_cache"])
    if policy_snapshot is not None:
        st["jpol"].load_snapshot(policy_snapshot)
    shared_analysis = _new_sibling_analysis_cache(
        st["model_b5"], st["single_head"], st["direct_head"])
    shared_replay = _new_replay_cache()
    out = []
    for kid in range(int(k)):
        _worker_heartbeat("trajectory_start", iid=root["iid"], traj_id=kid,
                          group_idx=int(group_idx))
        root_q, root_gid = _initial_root_stratum(
            root["seed"], root["iid"], root["episode_id"], root["state_hash"],
            kid, k, root.get("anchor_trajectories", 0))
        out.append(collect_trajectory_r14(
            st["jpol"], st["scorer"], st["executor"], st["model_b5"],
            st["single_head"], st["direct_head"], root["problem"],
            root["schedule"], root["root_ms"], root["iid"], root["episode_id"],
            root["progmem"],
            seed=_traj_seed(root["seed"], root["iid"], root["episode_id"],
                            root["state_hash"], kid),
            traj_id=kid, T=root["T"], eps=root["eps"], horizon=root["horizon"],
            step_offset=root["step_offset"], seed_rolex=root["seed_rolex"],
            seed_ast=root.get("seed_ast"), action_space=root["action_space"],
            val_cache=st["val_cache"],
            analyze_cache=shared_analysis, replay_cache=shared_replay,
            stop_on_negative=root.get("stop_on_negative", True),
            allow_policy_stop=root.get("allow_policy_stop", True),
            feasible_fallback=root.get("feasible_fallback", False),
            capture_visual_plan=root.get("capture_visual_plans", False),
            anchor_trajectories=root.get("anchor_trajectories", 0),
            initial_root_quantile=root_q, root_group_id=root_gid,
            initial_action_stratum=_initial_action_stratum(
                root["seed"], root["iid"], root["episode_id"], root["state_hash"],
                kid, k, root.get("anchor_trajectories", 0))))
        _worker_heartbeat("trajectory_end", iid=root["iid"], traj_id=kid,
                          group_idx=int(group_idx))
    _worker_heartbeat("job_end", iid=root["iid"],
                      siblings=int(k), group_idx=int(group_idx))
    return group_idx, out, {
        "worker_cpu_s": time.process_time() - cpu_started,
        "worker_wall_s": time.perf_counter() - wall_started,
        "cache_entries_before": cache_before,
        "cache_entries_after": len(st["val_cache"]),
        "cache_entries_added": len(st["val_cache"]) - cache_before,
        "n_siblings": int(k),
    }


def _mp_graph_shard_job_r14(job):
    """Collect a deterministic subset of one graph's sibling ids.

    Every shard loads the same frozen ``pi_old`` snapshot and uses the canonical
    per-sibling seed.  The parent reassembles ids 0..K-1 before group advantages
    are computed, so this changes scheduling only -- never trajectories or credit.
    """
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    group_idx, policy_snapshot, root, kids = job
    _worker_heartbeat("job_start", iid=root["iid"],
                      siblings=len(kids), group_idx=int(group_idx))
    st = _MP14_STATE
    if policy_snapshot is not None:
        st["jpol"].load_snapshot(policy_snapshot)
    if len(st["val_cache"]) > int(getattr(
            C, "T2L_WORKER_REPLAY_CACHE_ENTRIES", 1000)):
        st["val_cache"].clear()
    cache_before = len(st["val_cache"])
    shared_analysis = _new_sibling_analysis_cache(
        st["model_b5"], st["single_head"], st["direct_head"])
    shared_replay = _new_replay_cache()
    out = []
    for kid in kids:
        _worker_heartbeat("trajectory_start", iid=root["iid"], traj_id=kid,
                          group_idx=int(group_idx))
        root_q, root_gid = _initial_root_stratum(
            root["seed"], root["iid"], root["episode_id"], root["state_hash"],
            kid, root.get("branches", C.TO1_R13_K),
            root.get("anchor_trajectories", 0))
        tr = collect_trajectory_r14(
            st["jpol"], st["scorer"], st["executor"], st["model_b5"],
            st["single_head"], st["direct_head"], root["problem"],
            root["schedule"], root["root_ms"], root["iid"], root["episode_id"],
            root["progmem"],
            seed=_traj_seed(root["seed"], root["iid"], root["episode_id"],
                            root["state_hash"], kid),
            traj_id=kid, T=root["T"], eps=root["eps"], horizon=root["horizon"],
            step_offset=root["step_offset"], seed_rolex=root["seed_rolex"],
            seed_ast=root.get("seed_ast"), action_space=root["action_space"],
            val_cache=st["val_cache"],
            analyze_cache=shared_analysis, replay_cache=shared_replay,
            stop_on_negative=root.get("stop_on_negative", True),
            allow_policy_stop=root.get("allow_policy_stop", True),
            feasible_fallback=root.get("feasible_fallback", False),
            capture_visual_plan=root.get("capture_visual_plans", False),
            anchor_trajectories=root.get("anchor_trajectories", 0),
            initial_root_quantile=root_q, root_group_id=root_gid,
            initial_action_stratum=_initial_action_stratum(
                root["seed"], root["iid"], root["episode_id"], root["state_hash"],
                kid, root.get("branches", C.TO1_R13_K), root.get("anchor_trajectories", 0)))
        out.append((int(kid), tr))
        _worker_heartbeat("trajectory_end", iid=root["iid"], traj_id=kid,
                          group_idx=int(group_idx))
    _worker_heartbeat("job_end", iid=root["iid"],
                      siblings=len(kids), group_idx=int(group_idx))
    return group_idx, out, {
        "worker_cpu_s": time.process_time() - cpu_started,
        "worker_wall_s": time.perf_counter() - wall_started,
        "cache_entries_before": cache_before,
        "cache_entries_after": len(st["val_cache"]),
        "cache_entries_added": len(st["val_cache"]) - cache_before,
        "n_siblings": len(kids),
    }


def _compact_rollout_result_for_ipc(result):
    """Store large immutable rollout tensors as fp16 before worker->parent IPC.

    The actors cast these observations back to float32 before their trainable
    layers.  This halves the dominant serialized payload (proposal/root feature
    matrices) and is an opt-out stability trade-off via TRAIN_RECORD_FP16=0.
    Small state/context tensors and masks stay in their native dtype.
    """
    if not bool(getattr(C, "T2L_TRAIN_RECORD_FP16", False)):
        return result
    try:
        _group_idx, rows, _diag = result
        seen_m2 = set()
        for _kid, tr in rows:
            for rec in tr.get("steps", ()):
                for key in ("F_pool", "logits_old", "evid"):
                    value = rec.get(key)
                    if torch.is_tensor(value) and value.is_floating_point():
                        rec[key] = value.detach().to(dtype=torch.float16)
                m2 = rec.get("m2_rec")
                if not isinstance(m2, dict) or id(m2) in seen_m2:
                    continue
                seen_m2.add(id(m2))
                for key in ("cand_f", "cand_base", "cand_latent",
                            "cand_app_raw", "cand_memory"):
                    value = m2.get(key)
                    if torch.is_tensor(value) and value.is_floating_point():
                        m2[key] = value.detach().to(dtype=torch.float16)
        return result
    except Exception:
        # IPC compaction is an optimization only; never risk losing a rollout.
        return result


def _mp_packed_graph_job_r14(payload):
    """Cross the ProcessPool queue as one byte blob, never as tensor FDs.

    ``multiprocessing`` registers a special PyTorch tensor reducer that sends a
    separate shared-storage file descriptor for every tensor nested in a result.
    A 10-step rollout contains thousands of small tensors, which can break the
    resource-sharer channel with ``received 0 items of ancdata``.  Standard
    pickle runs inside the worker, so the outer process queue sees exactly one
    bytes object in each direction.  Numerical values and object structure are
    reconstructed unchanged in the parent.
    """
    use_sibling_shards, job = pickle.loads(payload)
    result = (_mp_graph_shard_job_r14(job) if use_sibling_shards
              else _mp_graph_job_r14(job))
    result = _compact_rollout_result_for_ipc(result)
    return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)


def _root_rollout_cost_hint(root):
    """Policy-independent LPT hint used only to submit expensive roots first."""
    problem = root["problem"]
    operations = list(getattr(problem, "operations", ()) or ())
    n_ops = len(operations)
    n_modes = sum(len(getattr(op, "modes", ()) or ()) for op in operations)
    seed_rolex = root.get("seed_rolex")
    n_prop = len(seed_rolex[1]) if seed_rolex is not None else 0
    # Proposal validation/continuation dominates; operation/mode counts capture
    # replay cost when two roots happen to expose similar proposal counts.
    return float(16 * n_prop + 2 * n_ops + n_modes)


def collect_depth_groups_rollouts_r14(
        jpol, scorer, executor, model_b5, single_head, direct_head, graphs,
        k=None, T=None, eps=None, horizon=None, seed=0, workers=1,
        step0_cache=None, mp_ctx=None, action_space="full", val_cache=None,
        critical_gate=False, torch_threads=1, rollout_pool=None,
        stop_on_negative=True, allow_policy_stop=True, feasible_fallback=False,
        progress_callback=None, anchor_trajectories=None):
    """Collect a complete rolling depth in parallel at graph granularity.

    The canonical loop already freezes one policy snapshot while collecting all
    active graphs and updates only after the depth.  Flattening graph x sibling
    jobs therefore changes scheduling overhead only, not seeds, trajectories,
    credit, graph order, or the GRPO update boundary.
    """
    k = int(k if k is not None else C.TO1_R13_K)
    t_batch0 = time.time()
    roots = []
    for g in graphs:
        h0 = schedule_hash(g.schedule)
        if step0_cache is not None:
            seed_rolex, seed_ast, seed_ast_s = _seed_state13(
                step0_cache, g.problem, g.schedule, g.iid)
        else:
            seed_rolex, seed_ast, seed_ast_s = None, None, 0.0
        roots.append({
            "problem": g.problem, "schedule": g.schedule,
            "root_ms": int(g.ms), "iid": g.iid,
            "episode_id": g.episode_id, "progmem": g.progmem,
            "state_hash": h0, "seed": int(seed), "T": T, "eps": eps,
            "horizon": horizon, "step_offset": g.gstep,
            "branches": int(k),
            "seed_rolex": seed_rolex, "action_space": action_space,
            "stop_on_negative": bool(stop_on_negative),
            "allow_policy_stop": bool(allow_policy_stop),
            "feasible_fallback": bool(feasible_fallback),
            "capture_visual_plans": bool(getattr(g, "capture_visual_plans", False)),
            "anchor_trajectories": (
                int(anchor_trajectories) if anchor_trajectories is not None else
                (min(max(0, int(getattr(C, "T2L_ANCHOR_TRAJECTORIES", 2))),
                     max(k - 2, 0))
                 if action_space == "policy_sampled" else 0)),
            "seed_ast": seed_ast, "seed_ast_s": seed_ast_s,
            "ast_breakdown": dict((seed_ast or {}).get("_profile", {})),
        })
    if not roots:
        return []

    # Serial remains the exact reference path used by parity tests and fallback.
    if int(workers) <= 1:
        return [collect_full_group_rollouts_r14(
            jpol, scorer, executor, model_b5, single_head, direct_head,
            r["problem"], r["schedule"], r["root_ms"], r["iid"],
            r["episode_id"], r["progmem"], k=k, T=T, eps=eps,
            horizon=horizon, seed=seed, step_offset=r["step_offset"], workers=1,
            step0_cache=step0_cache, mp_ctx=mp_ctx, action_space=action_space,
            val_cache=val_cache,
            stop_on_negative=bool(stop_on_negative),
            allow_policy_stop=bool(allow_policy_stop),
            feasible_fallback=bool(feasible_fallback),
            capture_visual_plans=bool(r["capture_visual_plans"])) for r in roots]

    # Freeze pi_old once.  Every task receives that same actor snapshot and no
    # update occurs until the whole depth has returned.  With enough active graphs
    # we retain one graph per task for cache locality.  At late depths, dynamically
    # shard K siblings so a handful of straggler graphs can still use the pool.
    # A fresh pool receives a deep-copied frozen policy in its initializer, so
    # serializing the same snapshot into every job is pure IPC/RAM duplication.
    # Keep snapshots only for an externally persistent pool that must be refreshed.
    policy_snapshot = jpol.snapshot() if rollout_pool is not None else None
    shard_specs = _sibling_shard_jobs(len(roots), k, workers)
    shards_per_graph = max((sum(1 for gi2, _ in shard_specs if gi2 == gi)
                            for gi in range(len(roots))), default=1)
    use_sibling_shards = bool(shards_per_graph > 1)
    if use_sibling_shards:
        jobs = [(gi, policy_snapshot, roots[gi], kids)
                for gi, kids in shard_specs]
    else:
        jobs = [(gi, policy_snapshot, root, k) for gi, root in enumerate(roots)]
    # Longest-processing-time-first submission reduces the final idle tail.  The
    # tagged results are restored to graph/sibling order below.
    jobs.sort(key=lambda job: _root_rollout_cost_hint(job[2]), reverse=True)
    # A Future can remain pending forever when a worker enters an extreme graph
    # analysis path or a native/PyTorch lock.  The former loop merely printed a
    # WAITING line every minute and had no terminal condition.  Use a per-cycle
    # pool plus a no-progress watchdog so one poisoned worker cannot waste a
    # multi-day run.  Retried jobs keep the same frozen policy and canonical
    # trajectory seeds, hence this changes execution scheduling only.
    from concurrent.futures import FIRST_COMPLETED, wait
    stall_timeout_s = max(
        60.0, float(os.environ.get("T2M_ROLLOUT_STALL_TIMEOUT_S", "900")))
    stall_retries = max(
        0, int(os.environ.get("T2M_ROLLOUT_STALL_RETRIES", "2")))
    if rollout_pool is not None:
        # External persistent pools cannot be safely replaced inside this call.
        # The formal T2-M runner now deliberately passes None.
        active_pool = rollout_pool
    else:
        active_pool = None

    tagged = []
    completed = 0
    original_job_count = len(jobs)
    remaining_jobs = list(jobs)
    retry_index = 0
    worker_budget = max(1, int(workers))
    while remaining_jobs:
        n_workers = max(1, min(worker_budget, len(remaining_jobs)))
        ex = active_pool or make_graph_rollout_pool(
            jpol, scorer, executor, model_b5, single_head, direct_head,
            workers=n_workers, mp_ctx=mp_ctx,
            val_cache=val_cache, critical_gate=critical_gate,
            torch_threads=torch_threads)
        forced_shutdown = False
        retry_jobs = None
        try:
            # Bound both parent-side pickled payloads and ProcessPool's internal
            # work queue.  Submitting every graph/shard at once duplicates large
            # roots and rollout metadata in RAM even though only `workers` jobs can
            # run.  A 2x worker window keeps CPUs fed without queue blow-up.
            queued = deque(remaining_jobs)
            pending = {}
            max_inflight = max(1, min(
                len(remaining_jobs), n_workers * max(1, int(getattr(
                    C, "T2L_MAX_INFLIGHT_MULTIPLIER", 2)))))

            def _refill():
                while queued and len(pending) < max_inflight:
                    job = queued.popleft()
                    payload = pickle.dumps(
                        (use_sibling_shards, job), protocol=pickle.HIGHEST_PROTOCOL)
                    future = ex.submit(_mp_packed_graph_job_r14, payload)
                    pending[future] = job

            _refill()
            last_progress = time.monotonic()
            while pending:
                done, _not_done = wait(
                    tuple(pending), timeout=60.0,
                    return_when=FIRST_COMPLETED)
                if not done:
                    pending_iids = sorted({str(job[2]["iid"])
                                           for job in pending.values()})
                    if progress_callback is not None:
                        progress_callback(
                            completed, original_job_count,
                            time.time() - t_batch0,
                            pending_iids=pending_iids, waiting=True)
                    stalled_for = time.monotonic() - last_progress
                    if stalled_for < stall_timeout_s:
                        continue
                    retry_jobs = list(pending.values()) + list(queued)
                    heartbeat = _worker_heartbeat_summary()
                    print(
                        "[t2m] collect STALL "
                        f"no_progress_s={stalled_for:.1f} "
                        f"completed_jobs={completed} "
                        f"pending_jobs={len(retry_jobs)} "
                        f"retry={retry_index}/{stall_retries} "
                        f"pending_instances={','.join(pending_iids[:16])} "
                        f"worker_stages={heartbeat}",
                        flush=True,
                    )
                    forced_shutdown = True
                    _force_shutdown_process_pool(ex)
                    break
                for future in done:
                    job = pending.pop(future)
                    try:
                        tagged.append(pickle.loads(future.result()))
                    except Exception as exc:
                        # Treat an abruptly terminated worker exactly like a
                        # stall: rebuild the pool and deterministically retry
                        # every unfinished/unqueued shard, including this one.
                        retry_jobs = [job] + list(pending.values()) + list(queued)
                        print(
                            "[t2m] collect WORKER_FAILURE "
                            f"type={type(exc).__name__} retry={retry_index}/"
                            f"{stall_retries} pending_jobs={len(retry_jobs)}",
                            flush=True,
                        )
                        forced_shutdown = True
                        _force_shutdown_process_pool(ex)
                        break
                    completed += 1
                    last_progress = time.monotonic()
                if forced_shutdown:
                    break
                _refill()
                if progress_callback is not None:
                    progress_callback(
                        completed, original_job_count,
                        time.time() - t_batch0,
                        pending_iids=None, waiting=False)
        finally:
            if not forced_shutdown and active_pool is None:
                ex.shutdown(wait=True)

        if not forced_shutdown:
            remaining_jobs = []
            break
        if retry_index >= stall_retries:
            stuck_iids = sorted({str(job[2]["iid"])
                                 for job in (retry_jobs or ())})
            raise RuntimeError(
                "rollout collection stalled after bounded retries; "
                f"instances={stuck_iids}; "
                f"worker_stages={_worker_heartbeat_summary()}")
        retry_index += 1
        remaining_jobs = list(retry_jobs or ())
        if active_pool is None and worker_budget > 1:
            # A broken worker pool is commonly cgroup OOM under large graph
            # rollouts.  Retrying at the same concurrency recreates the failure;
            # halve parallelism while keeping identical seeds/policy/reward.
            worker_budget = max(1, worker_budget // 2)
            print(f"[t2m] collect RECOVERY workers={worker_budget}", flush=True)
        # The second recovery attempt isolates each sibling.  A pathological
        # state then occupies at most one worker and its heartbeat identifies
        # the exact trajectory/step rather than hiding behind a six-sibling job.
        if retry_index >= 2 and use_sibling_shards:
            split_jobs = []
            for gi, snapshot, root, kids in remaining_jobs:
                split_jobs.extend(
                    (gi, snapshot, root, (int(kid),)) for kid in kids)
            remaining_jobs = split_jobs
        print(
            "[t2m] collect RECOVER "
            f"attempt={retry_index}/{stall_retries} "
            f"jobs={len(remaining_jobs)} singleton_shards="
            f"{bool(retry_index >= 2 and use_sibling_shards)}",
            flush=True,
        )
    batch_s = time.time() - t_batch0

    by_group = [[] for _ in roots]
    worker_telemetry = [[] for _ in roots]
    for gi, trajs_or_pairs, telemetry in tagged:
        if use_sibling_shards:
            by_group[gi].extend(trajs_or_pairs)
        else:
            by_group[gi].extend(enumerate(trajs_or_pairs))
        worker_telemetry[gi].append(telemetry)
    groups = []
    per_group_s = batch_s / max(len(roots), 1)
    for gi, root in enumerate(roots):
        ordered = sorted(by_group[gi], key=lambda item: item[0])
        got_ids = [kid for kid, _tr in ordered]
        if got_ids != list(range(k)):
            raise AssertionError(
                f"sibling shard coverage drift graph={gi}: {got_ids} != 0..{k-1}")
        trajs = [tr for _kid, tr in ordered]
        groups.append(_finish_group_r14(
            trajs, root["iid"], root["state_hash"], root["root_ms"],
            root["progmem"], per_group_s, n_workers,
            parallelism=("mp_sibling_sharded" if use_sibling_shards
                         else "mp_graph_persistent"), batch_coll_s=batch_s))
        tels = worker_telemetry[gi]
        groups[-1]["worker_telemetry"] = {
            "shards": len(tels),
            "worker_cpu_s": sum(float(t.get("worker_cpu_s", 0.0)) for t in tels),
            "worker_wall_s": max((float(t.get("worker_wall_s", 0.0))
                                  for t in tels), default=0.0),
            "worker_wall_sum_s": sum(float(t.get("worker_wall_s", 0.0)) for t in tels),
            "cache_entries_added": sum(int(t.get("cache_entries_added", 0)) for t in tels),
            "cache_entries_max": max((int(t.get("cache_entries_after", 0))
                                      for t in tels), default=0),
            "n_siblings": sum(int(t.get("n_siblings", 0)) for t in tels),
            "cost_hint": _root_rollout_cost_hint(root),
        }
        groups[-1]["ast_profile"] = {
            "ast_calls_before": int(k), "ast_calls_after": 1,
            "ast_seconds_per_call": float(root["seed_ast_s"]),
            "ast_seconds_saved": float(root["seed_ast_s"] * max(k - 1, 0)),
            "breakdown": dict(root["ast_breakdown"]),
        }
    return groups


# ---------------------------------------------------------------------------
# joint GRPO update -- stagewise OR shared credit; lambda_M2*L_M2 + lambda_M3*L_M3
# ---------------------------------------------------------------------------
def _m2_candidate_record(cand_f, cand_base, cand_latent=None, state_context=None,
                         cand_memory=None, cand_app_raw=None, cand_app_mask=None, cand_trace_graph=None):
    return {"feats": cand_f, "base": cand_base, "latents": cand_latent,
            "state_context": state_context, "memory_evidence": cand_memory,
            "app_raw": cand_app_raw, "app_mask": cand_app_mask,
            "trace_graph": cand_trace_graph,
            "n_cand": int(len(cand_base))}


def _m2_draw_logp_new(adapter, cand_f, cand_base, draw, T,
                      cand_latent=None, state_context=None, cand_memory=None,
cand_app_raw=None, cand_app_mask=None, cand_trace_graph=None):
    rem = torch.tensor(draw["rem_ids"], dtype=torch.long, device=cand_f.device)
    if hasattr(adapter, "final_logits_from_candidates"):
        cand = _m2_candidate_record(cand_f, cand_base, cand_latent, state_context,
                                    cand_memory, cand_app_raw, cand_app_mask, cand_trace_graph)
        logits = adapter.final_logits_from_candidates(cand, rem) / T
    else:
        delta = adapter.net(cand_f[rem].float())[:, 0]
        logits = (cand_base[rem].float() +
                  float(TO1_R13_ALPHA_M2) * torch.tanh(delta)) / T
    probs = torch.softmax(logits, dim=0)
    eps2 = float(C.TO1_R13_MIX_EPS)
    probs = (1.0 - eps2) * probs + eps2 / max(len(rem), 1)
    lp = torch.log(probs.clamp_min(1e-12))
    pos = rem.tolist().index(draw["idx"])
    return lp[pos]


def _m2_kl_ref(adapter, cand_f, cand_base, pool_mask, T,
               cand_latent=None, state_context=None, cand_memory=None,
cand_app_raw=None, cand_app_mask=None, cand_trace_graph=None):
    ridx = torch.tensor(pool_mask, dtype=torch.long, device=cand_f.device)
    if len(ridx) == 0:
        return torch.zeros((), dtype=torch.float32)
    if hasattr(adapter, "final_logits_from_candidates"):
        cand = _m2_candidate_record(cand_f, cand_base, cand_latent, state_context,
                                    cand_memory, cand_app_raw, cand_app_mask, cand_trace_graph)
        s_t = adapter.final_logits_from_candidates(cand, ridx) / T
    else:
        delta = adapter.net(cand_f[ridx].float())[:, 0]
        s_t = (cand_base[ridx].float() +
               float(TO1_R13_ALPHA_M2) * torch.tanh(delta)) / T
    s_r = cand_base[ridx].float() / T
    p_t = torch.softmax(s_t, dim=0)
    p_r = torch.softmax(s_r, dim=0)
    eps2 = float(C.TO1_R13_MIX_EPS)
    p_t = (1.0 - eps2) * p_t + eps2 / max(len(ridx), 1)
    p_r = (1.0 - eps2) * p_r + eps2 / max(len(ridx), 1)
    return (p_t * (torch.log(p_t) - torch.log(p_r))).sum()


def grpo_update_joint(jpol, groups, stage="C", T=None, eps=None, clip_eps=None,
                      seeds=None, epochs=None, log_prefix="[r13]", credit="shared",
                      optimizer_persistent=False):
    """One joint GRPO update on collected R13/R14 groups.

    credit:
      "shared"    (R13) -- a single per-trajectory A is shared by M2 draws and M3
                  actions (§25).  stage A/B/C semantics preserved verbatim.
      "stagewise" (R14) -- M2 draws take ONLY A2 (from local q2/U2, §14) and M3 steps
                  take ONLY A3 (terminal, §17); the §15 hard rule that M2 must not be
                  trained by M3's terminal advantage is enforced here.  per-system
                  informative gating (inf2 from U2 std, inf3 from R3 std).  M2
                  surrogate = per-draw clipped ratio mean, then per-state mean, then
                  per-trajectory mean (§19); M3 surrogate = R12 per-step clipped ratio
                  x A3, per-trajectory-equal mean (§20).

    L = lambda_M3*mean_traj(m3_sur) + lambda_M2*mean_traj(m2_sur)
        + beta_M3*mean_step KL_m3(ref=R6) + beta_M2*mean_state KL_m2(ref=attr prior).
    lambda_M2 = 1.0 under stagewise (§22), 0.5 under shared (R13).  One backward
    updates both systems (§21); KLs accumulate on all steps regardless of informativeness.
    """
    T = float(T if T is not None else C.TO1_R13_TEMP)
    eps = float(eps if eps is not None else C.TO1_R13_MIX_EPS)
    clip_eps = float(clip_eps if clip_eps is not None else C.TO1_R13_CLIP_EPS)
    epochs = int(epochs if epochs is not None else C.TO1_R13_UPDATE_EPOCHS)
    stagewise = bool(credit == "stagewise")
    lambda_m2 = float(C.TO1_R14_LAMBDA_M2 if stagewise else C.TO1_R13_LAMBDA_M2)
    train_m2 = stage in ("A", "C")
    train_m3 = stage in ("B", "C")
    lr_m3 = float(C.TO1_R13_LR_M3 if train_m3 else 0.0)
    lr_m2 = float(C.TO1_R13_LR_M2 if train_m2 else 0.0)

    def _bundles():
        bs = []
        n_steps = 0
        for g in groups:
            inf = bool(g.get("informative"))
            inf2 = bool(g.get("info2", inf))
            for tr in g.get("trajs", []):
                sts = tr["steps"]
                n_steps += len(sts)
                for rec in sts:
                    rec.setdefault("informative", inf)
                    rec.setdefault("inf3", inf if stagewise else inf)
                    rec.setdefault("inf2", inf2 if stagewise else inf)
                    rec.setdefault("adv_group", 0.0)
                    rec.setdefault("adv3", 0.0)
                    rec.setdefault("adv2", 0.0)
                bs.append((sts, inf, inf2))
        return bs, n_steps
    bundles, n_steps = _bundles()

    params_m3 = ([p for p in jpol.m3.parameters() if p.requires_grad]
                 if train_m3 else [])
    params_m2 = ([p for p in jpol.m2.parameters() if p.requires_grad]
                 if train_m2 else [])
    params = params_m3 + params_m2
    if not params:
        return {"n_informative_trajectories": 0, "n_trajectories": len(bundles),
                "n_steps": n_steps, "epochs": [], "reason": "no_trainable_params",
                "n_informative_trajectories_m2": 0, "final_kl_m3": None,
                "final_kl_m2": None}
    opt = getattr(jpol, "_t2d_optimizer", None) if optimizer_persistent else None
    current_ids = tuple(id(p) for p in params)
    if opt is None or getattr(jpol, "_t2d_optimizer_param_ids", ()) != current_ids:
        if optimizer_persistent:
            # T2-D keeps independent actor learning rates and Adam moments.  The
            # non-persistent branch below intentionally retains the frozen P0/T2-B
            # baseline's original single-group optimizer semantics.
            param_groups = []
            if params_m2:
                param_groups.append({"params": params_m2, "lr": lr_m2,
                                     "actor": "m2"})
            if params_m3:
                param_groups.append({"params": params_m3, "lr": lr_m3,
                                     "actor": "m3"})
            opt = torch.optim.AdamW(param_groups, weight_decay=0.0)
        else:
            opt = torch.optim.AdamW(params, lr=lr_m3, weight_decay=0.0)
        if optimizer_persistent:
            jpol._t2d_optimizer = opt
            jpol._t2d_optimizer_param_ids = current_ids
            resume_state = getattr(jpol, "_t2d_resume_optimizer_state", None)
            if resume_state is not None:
                opt.load_state_dict(resume_state)
                delattr(jpol, "_t2d_resume_optimizer_state")
    torch.manual_seed(0 if seeds is None else int(seeds))

    n_info_traj = sum(1 for sts, inf, _i2 in bundles if inf and len(sts))
    count_inf2 = sum(1 for sts, _i3, inf2 in bundles if inf2 and len(sts))
    per_epoch = []
    for ep in range(epochs):
        sur_trajs = []
        s3_trajs, s2_trajs = [], []
        kl_ref_m3, kl_old = [], []
        kl_m2 = []
        n_clip_m3 = 0
        for sts, inf_g, inf2_g in bundles:
            if not sts:
                continue
            s3 = []
            s2 = []
            for rec in sts:
                if stagewise:
                    A3, A2 = float(rec["adv3"]), float(rec["adv2"])
                    g3, g2 = bool(rec["inf3"]), bool(rec["inf2"])
                else:
                    A3 = A2 = float(rec["adv_group"])
                    g3 = g2 = inf_g
                if train_m3:
                    logits = jpol.m3.action_logits(
                        rec["F_pool"], rec["sf_t"], rec["pool_stats"],
                        evid=rec.get("evid"), traj_ctx=rec.get("traj_ctx"),
                        pair_mask=rec.get("pair_mask"))
                    stop_allowed = bool(rec.get("stop_allowed", True))
                    train_logits = logits if stop_allowed else logits[:-1]
                    base_logits = jpol.m3.base_logits(
                        rec["F_pool"], rec["sf_t"], rec["pool_stats"],
                        evid=rec.get("evid"), traj_ctx=rec.get("traj_ctx"),
                        pair_mask=rec.get("pair_mask"))
                    old_logits = rec["logits_old"]
                    if not stop_allowed:
                        base_logits = base_logits[:-1]
                        old_logits = old_logits[:-1]
                    lp_new = mixture_logp(train_logits, rec["a"], T, eps)
                    ratio = torch.exp(lp_new - rec["logp_old"])
                    if g3:
                        s3.append(torch.min(ratio * A3, torch.clamp(
                            ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A3))
                        if bool((ratio > 1.0 + clip_eps).item() or
                                (ratio < 1.0 - clip_eps).item()):
                            n_clip_m3 += 1
                    kl_ref_m3.append(_kl_mixture(
                        train_logits, base_logits, T, eps))
                    kl_old.append(_kl_mixture(train_logits, old_logits, T, eps))
                if train_m2 and rec.get("m2_rec") is not None:
                    m2 = rec["m2_rec"]
                    c_f = m2["cand_f"]
                    c_b = m2["cand_base"]
                    step_s2 = []
                    draws = list(m2["draws"])
                    if (draws and m2.get("selection_semantics") ==
                            "ordered_without_replacement_root_set"):
                        # The complete Top-K root set is one hierarchical action.
                        # Its exact behavior probability is the product of the
                        # sequential conditional draw probabilities; summing log-p
                        # keeps the PPO ratio mathematically consistent and avoids
                        # rewarding K roots as if they were K independent actions.
                        lp_new_parts = [
                            _m2_draw_logp_new(
                                jpol.m2, c_f, c_b, d, float(C.TO1_R13_TEMP_M2),
                                m2.get("cand_latent"), m2.get("state_context"),
                                m2.get("cand_memory"), m2.get("cand_app_raw"),
                                m2.get("cand_app_mask"), m2.get("cand_trace_graph"))
                            for d in draws
                        ]
                        lp_new_joint = torch.stack(lp_new_parts).sum()
                        lp_old_joint = sum(float(d["logp"]) for d in draws)
                        ratio = torch.exp(lp_new_joint - lp_old_joint)
                        if g2:
                            step_s2.append(torch.min(
                                ratio * A2,
                                torch.clamp(ratio, 1.0 - clip_eps,
                                            1.0 + clip_eps) * A2))
                    else:
                        # Legacy/single-root gates retain their original behavior.
                        for d in draws:
                            lp_new = _m2_draw_logp_new(
                                jpol.m2, c_f, c_b, d, float(C.TO1_R13_TEMP_M2),
                                m2.get("cand_latent"), m2.get("state_context"),
                                m2.get("cand_memory"), m2.get("cand_app_raw"),
                                m2.get("cand_app_mask"), m2.get("cand_trace_graph"))
                            ratio = torch.exp(lp_new - d["logp"])
                            if g2:
                                step_s2.append(torch.min(
                                    ratio * A2, torch.clamp(
                                        ratio, 1.0 - clip_eps,
                                        1.0 + clip_eps) * A2))
                    if g2 and step_s2:
                        s2.append(sum(step_s2) / len(step_s2))
                    kl_m2.append(_m2_kl_ref(
                        jpol.m2, c_f, c_b, m2["pool_mask"],
                        float(C.TO1_R13_TEMP_M2), m2.get("cand_latent"),
                        m2.get("state_context"), m2.get("cand_memory"),
                        m2.get("cand_app_raw"), m2.get("cand_app_mask"), m2.get("cand_trace_graph")))
            if stagewise:
                if train_m3 and s3:
                    s3_trajs.append(sum(s3) / len(s3))
                if train_m2 and s2:
                    s2_trajs.append(sum(s2) / len(s2))
            elif inf_g:
                t_sur = 0.0
                if train_m3 and s3:
                    t_sur += sum(s3) / len(s3)
                if train_m2 and s2:
                    t_sur += lambda_m2 * (sum(s2) / len(s2))
                sur_trajs.append(t_sur)
        if stagewise:
            if not s3_trajs and not s2_trajs:
                per_epoch.append({"epoch": ep, "loss": 0.0,
                                  "n_informative_trajectories": 0, "n_steps": n_steps,
                                  "clip_frac_m3": 0.0, "kl_ref_m3": 0.0, "kl_m2": 0.0,
                                  "reason": "no_informative_trajectories",
                                  "stopped_stale": False})
                break
            loss = torch.zeros((), dtype=torch.float32)
            if train_m3 and s3_trajs:
                loss = loss - (sum(s3_trajs) / len(s3_trajs))
            if train_m2 and s2_trajs:
                loss = loss - lambda_m2 * (sum(s2_trajs) / len(s2_trajs))
        else:
            if not sur_trajs:
                per_epoch.append({"epoch": ep, "loss": 0.0,
                                  "n_informative_trajectories": 0, "n_steps": n_steps,
                                  "clip_frac_m3": 0.0, "kl_ref_m3": 0.0, "kl_m2": 0.0,
                                  "reason": "no_informative_trajectories",
                                  "stopped_stale": False})
                break
            loss = -(sum(sur_trajs) / len(sur_trajs))
        if train_m3 and kl_ref_m3:
            loss = loss + float(C.TO1_R14_BETA_M3 if stagewise else C.TO1_R13_BETA_M3) * (
                sum(kl_ref_m3) / len(kl_ref_m3))
        if train_m2 and kl_m2:
            loss = loss + float(C.TO1_R14_BETA_M2 if stagewise else C.TO1_R13_BETA_M2) * (
                sum(kl_m2) / len(kl_m2))
        opt.zero_grad()
        loss.backward()
        def _group_grad_norm(group):
            vals = [p.grad.detach().float().norm(2) ** 2 for p in group
                    if p.grad is not None]
            return float(torch.sqrt(sum(vals)).detach()) if vals else 0.0
        grad_norm_m2 = _group_grad_norm(params_m2)
        grad_norm_m3 = _group_grad_norm(params_m3)
        grad_norm = float(nn.utils.clip_grad_norm_(params, 10.0))
        opt.step()
        m_kl_m3 = float(sum(k.detach() for k in kl_ref_m3) / len(kl_ref_m3)) if kl_ref_m3 else 0.0
        m_kl_m2 = float(sum(k.detach() for k in kl_m2) / len(kl_m2)) if kl_m2 else 0.0
        per_epoch.append({
            "epoch": ep, "loss": float(loss.detach()),
            "n_informative_trajectories": (count_inf2 if stagewise else n_info_traj),
            "n_steps": n_steps,
            "n_info_steps": (count_inf2 if stagewise else n_info_traj),
            "grad_norm": grad_norm, "kl_ref_m3": m_kl_m3, "kl_old": (
                float(sum(k.detach() for k in kl_old) / len(kl_old)) if kl_old else 0.0),
            "kl_m2": m_kl_m2, "clip_frac_m3": float(n_clip_m3) / max(n_info_traj, 1),
            "n_informative_trajectories_m2": count_inf2,
            "grad_norm_m2": grad_norm_m2, "grad_norm_m3": grad_norm_m3,
            "stopped_stale": False, "reason": ""})
    return {"n_informative_trajectories": n_info_traj, "n_trajectories": len(bundles),
            "n_steps": n_steps, "epochs": per_epoch,
            "n_informative_trajectories_m2": count_inf2,
            "optimizer_persistent": bool(optimizer_persistent),
            "optimizer_state_entries": len(opt.state),
            "optimizer_param_groups": [g.get("actor") for g in opt.param_groups],
            "final_kl_m3": per_epoch[-1]["kl_ref_m3"] if per_epoch else None,
            "final_kl_m2": per_epoch[-1]["kl_m2"] if per_epoch else None}


# ---------------------------------------------------------------------------
# stochastic real advancement under the joint policy (gate active)  (§22-23)
# ---------------------------------------------------------------------------
def advance_step_r13(jpol, scorer, executor, model_b5, single_head, direct_head,
                     problem, schedule, iid, episode_id, progmem, root_ms,
                     step_offset=0, rng=None, T=None, eps=None, gate_variant="r13",
                     action_space="full"):
    """Sample ONE M3 action under the §0 gated pool and execute it (exploration).
    True-STOP semantics: STOP / infeasible / non-positive / no-pool / no-proposals.
    Mirrors R12 `advance_step_r12` with the M2 gate inserted between Appearance and
    the M3 pool.  gate_variant="r14" uses the ADAPTIVE probe + q2 gate so R18 train
    and collect share the identical structure; action_space="validated" (R18 §12-13)
    samples from the REAL counterfactual-validated action set with the evid
    observation, same as collect_full_group_rollouts_r14."""
    T = float(T if T is not None else C.TO1_R13_TEMP)
    eps = float(eps if eps is not None else C.TO1_R13_MIX_EPS)
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
    rng = rng if rng is not None else random.Random(
        int(_traj_seed(0, iid, episode_id, h, 0)))
    _gatefn = _m2_gate_step_r14 if gate_variant == "r14" else _m2_gate_step
    gate = _gatefn(ast, metas, prop_feats, jpol.m2, executor, progmem, iid,
                   episode_id, step_offset, sf, rng)
    if not gate["gated_metas"]:
        return {"action": "stop", "reason": "no_pool_m2", "state_hash": h,
                "m2_diag": gate["diag"]}
    rolex = _rollex_of(ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(progmem.features(iid, episode_id, step_offset, sf, queries),
                       dtype=torch.float32)
    mem = mem * float(progmem.retrieval_gate(iid, episode_id, step_offset, sf))
    evid = None
    if action_space == "validated":
        vs = build_validated_action_set_r18(
            ast, gate, rolex, scorer, executor, progmem, iid, episode_id,
            step_offset, sf, h, ms_cur, mem)
        if not vs["pool"]:
            return {"action": "stop", "reason": "no_validated_pool", "state_hash": h,
                    "m2_diag": gate["diag"]}
        pool = vs["pool"]
        F_pool = vs["F_pool"]
        pool_stats = vs["pool_stats"]
        evid = vs["evid"]
    elif action_space == "multistep":
        # R19 §6-24: H-step-validated action set (G1 -> GH -> Memory tiers).
        vs = build_multistep_action_set_r19(
            ast, gate, rolex, scorer, executor, progmem, iid, episode_id,
            step_offset, sf, h, ms_cur, mem,
            cont_m2=getattr(jpol, "cont_m2", None),
            cont_m3=getattr(jpol, "cont_m3", None),
            acache=cache, rc=None, hcache=None)
        if not vs["pool"]:
            return {"action": "stop", "reason": "no_multistep_pool", "state_hash": h,
                    "m2_diag": gate["diag"]}
        pool = vs["pool"]
        F_pool = vs["F_pool"]
        pool_stats = vs["pool_stats"]
        evid = vs["evid"]
    elif action_space == "lexicographic":
        # R20 §4-12: state-level lexicographic fallback (PA >> PB >> PC >> STOP).
        vs = build_lexicographic_action_set_r20(
            ast, gate, rolex, scorer, executor, progmem, iid, episode_id,
            step_offset, sf, h, ms_cur, mem,
            cont_m2=getattr(jpol, "cont_m2", None),
            cont_m3=getattr(jpol, "cont_m3", None),
            acache=cache, rc=None, hcache=None)
        if not vs["pool"]:
            return {"action": "stop", "reason": "no_lexicographic_pool", "state_hash": h,
                    "m2_diag": gate["diag"]}
        pool = vs["pool"]
        F_pool = vs["F_pool"]
        pool_stats = vs["pool_stats"]
        evid = vs["evid"]
    else:
        logit_pos, rank = _scores(scorer, rolex, mem)
        pool, pinfo = wide_pool(rolex, logit_pos, rank)
    out = {"action": "stop", "reason": "no_pool_wide", "state_hash": h,
           "m2_diag": gate["diag"]}
    if not pool:
        return out
    if action_space not in ("validated", "multistep"):
        F_pool = _rerank_feats_all(scorer, rolex, mem)[pool]
        pool_stats = _pool_stats_from(F_pool)
    traj_ctx = trajectory_context(
        step_index=step_offset, horizon=C.TO1_R13_HORIZON,
        root_makespan=root_ms, current_makespan=ms_cur, last_step_gain=0.0,
        action_count=step_offset, non_improving_streak=0)
    with torch.no_grad():
        logits = jpol.m3.action_logits(F_pool, sf_t, pool_stats, evid=evid,
                                       traj_ctx=traj_ctx)
    a = mixture_sample(logits, T, eps, rng)
    M = len(pool)
    out.update({"m2_diag": gate["diag"], "logits": logits.detach().float(), "M": M,
                "best_pool_score": float(logits[:M].max()),
                "stop_score": float(logits[-1])})
    if a == M:
        out.update({"action": "stop", "reason": "policy_stop", "state_hash": h})
        return out
    meta = gate["gated_metas"][pool[a]]
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


def advance_graphs_r13(jpol, scorer, executor, model_b5, single_head, direct_head,
                       graphs, rng, T=None, eps=None, log_prefix="[r13]",
                       max_depth=None, gate_variant="r13", action_space="full"):
    from .rolling_grpo import Graph  # noqa: F401  (Graph contract reused)
    max_depth = int(max_depth if max_depth is not None else C.TO1_R13_MAX_DEPTH)
    out = []
    for g in graphs:
        if g.done:
            out.append({"iid": g.iid, "action": "skip", "reason": "done"})
            continue
        grng = random.Random(int(rng.randint(0, 2 ** 31)))
        step = advance_step_r13(jpol, scorer, executor, model_b5, single_head,
                                direct_head, g.problem, g.schedule, g.iid, g.episode_id,
                                g.progmem, g.ms, step_offset=g.gstep, rng=grng,
                                T=T, eps=eps, gate_variant=gate_variant,
                                action_space=action_space)
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
                "fine_key": ((step["meta_type"], step["meta_role"], step["src"],
                              step["tgt"]) if step["meta_type"] == "single"
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
            g.done = True
            g.adv_reason = step["reason"]
            out.append({"iid": g.iid, "action": "stop", "reason": step["reason"],
                        "state_hash": step["state_hash"]})
    return out


# ---------------------------------------------------------------------------
# rolling-cycle runner for Stages A / B / C  (§32-36)
# ---------------------------------------------------------------------------
def _m2_probe_stats(groups):
    """Aggregate per-state M2 gate diagnostics across collected steps."""
    agg = {"n_states": 0, "probed_root_count": 0.0, "tier_A": 0.0, "tier_B": 0.0,
           "tier_C": 0.0, "positive_probe_rate": 0.0, "coverage_ratio": 0.0,
           "best_probe_gain_mean": 0.0, "n_pos_probe_states": 0}
    for g in groups:
        for tr in g["trajs"]:
            for rec in tr["steps"]:
                d = rec.get("m2_diag")
                if not d:
                    continue
                agg["n_states"] += 1
                pr = max(d["probed_root_count"], 1)
                agg["probed_root_count"] += d["probed_root_count"]
                agg["tier_A"] += d["tier_A"]
                agg["tier_B"] += d["tier_B"]
                agg["tier_C"] += d["tier_C"]
                agg["positive_probe_rate"] += d["positive_probe_count"] / pr
                agg["coverage_ratio"] += (d["gated_proposal_count"] /
                                          max(d["full_proposal_count"], 1))
                agg["best_probe_gain_mean"] += d["best_probe_gain"]
                if d["positive_probe_count"] > 0:
                    agg["n_pos_probe_states"] += 1
    n = max(agg["n_states"], 1)
    for key in ("probed_root_count", "tier_A", "tier_B", "tier_C",
                "positive_probe_rate", "coverage_ratio", "best_probe_gain_mean"):
        agg[key] = float(agg[key]) / n
    return agg


def _m2_prob_move(jpol, groups):
    """Mean |logp(theta)| - logp_old| over policy draws at CURRENT params (movement)."""
    tot, acc = 0, 0.0
    for g in groups:
        for tr in g["trajs"]:
            for rec in tr["steps"]:
                m2 = rec.get("m2_rec")
                if not m2:
                    continue
                for d in m2["draws"]:
                    lp = float(_m2_draw_logp_new(
                        jpol.m2, m2["cand_f"], m2["cand_base"], d,
                        float(C.TO1_R13_TEMP_M2), m2.get("cand_latent"),
                        m2.get("state_context"), m2.get("cand_memory"),
                        m2.get("cand_app_raw"),
                        m2.get("cand_app_mask"), m2.get("cand_trace_graph")).detach().item())
                    acc += abs(lp - d["logp"])
                    tot += 1
    return float(acc / tot) if tot else 0.0


def _cycle_reward_metrics(groups):
    """Descriptive terminal-reward metrics; never used by policy selection."""
    rewards = [float(r) for g in groups for r in g.get("rewards", [])]
    best_per_graph = [max(map(float, g.get("rewards", [])))
                      for g in groups if g.get("rewards")]
    if not rewards:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0,
                "max": 0.0, "positive_rate": 0.0, "negative_rate": 0.0,
                "zero_rate": 0.0, "mean_best_per_graph": 0.0,
                "informative_group_rate": 0.0, "n": 0}
    arr = np.asarray(rewards, dtype=np.float64)
    return {
        "mean": float(arr.mean()), "median": float(np.median(arr)),
        "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max()),
        "positive_rate": float(np.mean(arr > 0)),
        "negative_rate": float(np.mean(arr < 0)),
        "zero_rate": float(np.mean(arr == 0)),
        "mean_best_per_graph": float(np.mean(best_per_graph)) if best_per_graph else 0.0,
        "informative_group_rate": float(np.mean(
            [bool(g.get("informative")) for g in groups])) if groups else 0.0,
        "n": int(arr.size),
    }


def _depth_runtime_metrics(groups, collect_s, workers, active_graphs):
    """Observation-only rollout utilization/straggler/cache diagnostics."""
    tels = [g.get("worker_telemetry") or {} for g in groups]
    graph_wall = [float(t.get("worker_wall_s", 0.0)) for t in tels]
    graph_mean = float(np.mean(graph_wall)) if graph_wall else 0.0
    graph_max = max(graph_wall, default=0.0)
    cpu_s = sum(float(t.get("worker_cpu_s", 0.0)) for t in tels)
    lookups = sum(int(g.get("validation_cache_lookups", 0)) for g in groups)
    hits = sum(int(g.get("validation_cache_hits", 0)) for g in groups)
    prof_keys = ("analyze", "execute", "mem", "policy", "m2gate")
    prof = {key: sum(float(g.get("prof_cpu_s", {}).get(key, 0.0)) for g in groups)
            for key in prof_keys}
    return {
        "active_graphs": int(active_graphs),
        "submitted_shards": int(sum(int(t.get("shards", 1)) for t in tels)),
        "mean_graph_wall_s": graph_mean,
        "max_graph_wall_s": graph_max,
        "straggler_ratio": graph_max / max(graph_mean, 1e-12),
        "aggregate_worker_cpu_s": cpu_s,
        "effective_worker_utilization": cpu_s / max(float(collect_s) * max(int(workers), 1),
                                                     1e-12),
        # Historical field ``vprobe_ms`` actually stores elapsed seconds.  It is
        # summed over sibling trajectories, so expose it as aggregate validation
        # time rather than pretending it is a per-depth wall clock or CPU timer.
        "validation_seconds_aggregate": sum(
            float(g.get("vprobe_ms", 0.0)) for g in groups),
        "cache_lookups": int(lookups),
        "cache_hits": int(hits),
        "cache_hit_rate": hits / max(lookups, 1),
        "cache_entries_added": int(sum(int(t.get("cache_entries_added", 0)) for t in tels)),
        "cache_entries_max_observed": max((int(t.get("cache_entries_max", 0))
                                           for t in tels), default=0),
        "profile_cpu_s": prof,
        "slowest_graphs": sorted(
            [{"iid": g.get("iid"),
              "wall_s": float((g.get("worker_telemetry") or {}).get(
                  "worker_wall_s", 0.0)),
              "shards": int((g.get("worker_telemetry") or {}).get("shards", 1)),
              "cache_hit_rate": float(g.get("validation_cache_hit_rate", 0.0))}
             for g in groups], key=lambda row: row["wall_s"], reverse=True)[:5],
    }


@torch.no_grad()
def _t2d_actor_metrics(jpol, groups):
    m2_ent, m2_res, m2_ratio, m2_sizes = [], [], [], []
    m3_ent, m3_res, m3_ratio, m3_sizes, stop_ps = [], [], [], [], []
    selection_miss = []
    for g in groups:
        for tr in g.get("trajs", []):
            for rec in tr.get("steps", []):
                F, sf, ps = rec["F_pool"], rec["sf_t"], rec["pool_stats"]
                z = jpol.m3.action_logits(F, sf, ps, evid=rec.get("evid"),
                                          traj_ctx=rec.get("traj_ctx"),
                                          pair_mask=rec.get("pair_mask"))
                zr = jpol.m3.base_logits(F, sf, ps, evid=rec.get("evid"),
                                         traj_ctx=rec.get("traj_ctx"),
                                         pair_mask=rec.get("pair_mask"))
                p = torch.softmax(z / float(C.TO1_R13_TEMP), -1)
                d = z - zr
                m3_ent.append(float(-(p * torch.log(p.clamp_min(1e-12))).sum()))
                m3_res.append(float(d.norm()))
                m3_ratio.append(float(d.abs().mean() / (zr.abs().mean() + 1e-6)))
                m3_sizes.append(float(len(F)))
                stop_ps.append(float(p[-1]))
                evid = rec.get("evid")
                if evid is not None and len(evid) and evid.shape[-1] >= 1:
                    has_positive = bool((evid[:, 0] > 0).any())
                    a = int(rec.get("a", len(F)))
                    selected_positive = a < len(F) and float(evid[a, 0]) > 0.0
                    if has_positive:
                        selection_miss.append(float(not selected_positive))
                m2 = rec.get("m2_rec")
                if not m2:
                    continue
                ids = m2.get("pool_mask", [])
                if not ids:
                    continue
                ridx = torch.as_tensor(ids, dtype=torch.long)
                base = m2["cand_base"][ridx]
                if hasattr(jpol.m2, "final_logits_from_candidates"):
                    cand = _m2_candidate_record(
                        m2["cand_f"], m2["cand_base"], m2.get("cand_latent"),
                        m2.get("state_context"), m2.get("cand_memory"),
                        m2.get("cand_app_raw"), m2.get("cand_app_mask"),
                        m2.get("cand_trace_graph"))
                    final = jpol.m2.final_logits_from_candidates(cand, ridx)
                else:
                    final = base + float(C.TO1_R13_ALPHA_M2) * torch.tanh(
                        jpol.m2.net(m2["cand_f"][ridx])[:, 0])
                pp = torch.softmax(final / float(C.TO1_R13_TEMP_M2), -1)
                dd = final - base
                m2_ent.append(float(-(pp * torch.log(pp.clamp_min(1e-12))).sum()))
                m2_res.append(float(dd.norm()))
                m2_ratio.append(float(dd.abs().mean() / (base.abs().mean() + 1e-6)))
                m2_sizes.append(float(len(ids)))
    avg = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {"m2/entropy": avg(m2_ent), "m2/residual_norm": avg(m2_res),
            "m2/residual_to_base_ratio": avg(m2_ratio),
            "m2/root_set_size": avg(m2_sizes),
            "m3/entropy": avg(m3_ent), "m3/residual_norm": avg(m3_res),
            "m3/residual_to_base_ratio": avg(m3_ratio),
            "m3/proposal_set_size": avg(m3_sizes),
            "m3/stop_probability": avg(stop_ps),
            "m3/selection_miss": avg(selection_miss)}


def _r13_cycle_picks(schedule_specs, cycles, per_src, seed, stage,
                     unified=False, per_cycle=None):
    """Seeded round-robin graph draw per cycle.

    Draws NOT per 3x-episode-expanded queue slot but by cycling DISTINCT instances, so
    a cycle never benches the same instance twice and aux graphs spread across cycles.
    Deterministic: same (seed, stage) reproduces the identical assignment.  Out of the
    original per-cycle queue: bench[4],[5] were the same slot-pair at every stage start,
    so a batch whose instance pair yields no positive root probes went fully dry and the
    Stage-A "not declining" gates fired on a sampling miss rather than a regression.

    unified=True (R21 D1): ignore per_src source quotas and draw `per_cycle` DISTINCT
    instances per cycle from ONE shuffled pool over bench+real+syn -- a genuinely
    instance-balanced distribution (equal draw probability, no source over-sampling).

    Fresh episodes are guaranteed by the runner calling g.reset() at the START of every
    cycle (a graph shared across cycles is then pristine when its cycle runs -- the
    build-time reset was too early once an earlier cycle had advanced it to done).

    Returns {cycle: [graph]} (pure: no graph mutation here).
    """
    _draw_seed = int(seed) + {"A": 0xA13, "B": 0xB13, "C": 0xC13}.get(stage, 0xD13)
    _draw_rng = random.Random(_draw_seed)
    pools = {}
    for src, grs in (("bench", schedule_specs["bench"]),
                     ("real", schedule_specs["real"]),
                     ("syn", schedule_specs["syn"])):
        _seen_ids, uniq = set(), []
        for _g in grs:
            if _g is None or id(_g) in _seen_ids:
                continue
            _seen_ids.add(id(_g))
            uniq.append(_g)
        uniq.sort(key=lambda _g: (getattr(_g, "iid", ""), str(id(_g))))
        _draw_rng.shuffle(uniq)
        pools[src] = uniq
    cycle_handle = {c: [] for c in range(cycles)}
    if unified:
        full = []
        for src in ("bench", "real", "syn"):
            full.extend(pools[src])
        _draw_rng.shuffle(full)
        pc = int(per_cycle if per_cycle is not None else sum(per_src.values()))
        pick = 0
        for c in range(cycles):
            for _ in range(pc):
                if not full:
                    break
                cycle_handle[c].append(full[pick % len(full)])
                pick += 1
        return cycle_handle
    _pick_idx = {src: 0 for src in pools}
    for c in range(cycles):
        for src in ("bench", "real", "syn"):
            for _ in range(per_src.get(src, 0)):
                pool = (pools[src] or pools["bench"] or [])
                if not pool:
                    continue
                g = pool[_pick_idx[src] % len(pool)]
                _pick_idx[src] += 1
                cycle_handle[c].append(g)
    return cycle_handle


def _stage_a_gates_summary(base, non_dry_stats, stage_grad_steps, stage_prob_move):
    """Stage-A PASS gates at the STAGE level, computed once over non-dry cycles.

    "Not declining" is judged on the STAGE AGGREGATE (mean over non-dry cycles) vs the
    pre-training base (first non-dry cycle's stats).  Per-batch positive-probe yield
    swings several-fold between adjacent 4-graph batches (measured 0.10->0.57), so any
    adjacent-cycle comparison false-fails on noise; a GENUINE collapse tozero shows up
    as an all-dry stage or a near-zero mean and still fails.  Movement gates reflect
    the stage's cumulative M2 parameter update (grad steps / prob movement actually
    observed), so an all-dry stage (no update ever) fails regardless.

    Returns the 5-gate dict (m2_grad_nonzero / prob_move / positive_probe_rate_ok /
    tier_a_coverage_ok / gated_coverage_ok) + n_non_dry_cycles + all_dry.
    """
    n = len(non_dry_stats)
    if n == 0:
        return {"m2_grad_nonzero": bool(stage_grad_steps > 0),
                "prob_move": float(stage_prob_move) > 1e-4,
                "positive_probe_rate_ok": False, "tier_a_coverage_ok": False,
                "gated_coverage_ok": False, "n_non_dry_cycles": 0, "all_dry": True}
    rate = float(np.mean([s["positive_probe_rate"] for s in non_dry_stats]))
    tier = float(np.mean([s["tier_A"] for s in non_dry_stats]))
    cov = float(np.mean([s["coverage_ratio"] for s in non_dry_stats]))
    return {
        "m2_grad_nonzero": bool(stage_grad_steps > 0),
        "prob_move": float(stage_prob_move) > 1e-4,
        "positive_probe_rate_ok": bool(rate >= 0.5 * base["positive_probe_rate"] - 0.02
                                       and rate >= 0.02),
        "tier_a_coverage_ok": bool(tier >= 0.5 * base["tier_A"] - 0.05 and tier >= 0.5),
        "gated_coverage_ok": bool(cov >= 0.5 * base["coverage_ratio"] - 0.05
                                  and cov >= 0.02),
        "n_non_dry_cycles": int(n), "all_dry": False,
    }


def run_rolling_cycles_r13(jpol, scorer, env, schedule_specs, stage="C", cycles=None,
                           k=None, horizon=None, graphs_per_batch=None, workers=1,
                           seed=0, quick=False, log_prefix="[r13]",
                           eval_root_builder=None, collapse_floor=None,
                           parent_policy=None, mp_ctx=None, adapt_eta=None,
                           variant="r13", on_groups=None, action_space="full",
                           per_src=None, unified=False, per_cycle=None,
                           best_score_fn=None, depth_parallel=True,
                           worker_critical_gate=False, worker_torch_threads=1,
                           optimizer_persistent=False, tensorboard_writer=None,
                           rollout_pool=None):
    """Rolling GRPO over the §0 agentic loop.

    stage A: M2 adapter trains on FROZEN M3; PASS gates measured per cycle (M2 grad,
    prob movement, positive-root probe rate not declining, Tier-A coverage not
    declining, normal-M5 path preserved -- the last checked externally via the
    always-on evaluator's state coverage so no train untested).
    stage B: M2 frozen; M3 rolling GRPO on the GATED pool (R12 semantics).
    stage C: JOINT M2+M3 (one optimizer, two param groups).  R14 runs its ONE joint
    stage as stage "C" with variant="r14" (§2): the collect uses the ADAPTIVE probe +
    q2 (collect_full_group_rollouts_r14) and the update uses stagewise A2/A3 credit.

    variant: "r13" | "r14" -- select the collect/credit regime.
    on_groups: optional callback(flat list of collected groups) invoked after each
    depth's collect batch (used by R14 §34 reward diagnostics).

    Per cycle: for each depth, collect K sibling trajectories per active graph via
    ProcessPoolExecutor -> grpo_update_joint -> stochastic real advancement
    (`advance_graphs_r13`). Same `Graph` rotation, collapse guard on the unified-ruler
    TRAIN gain, and best-cycle rollback discipline as R12.
    """
    use_r14 = bool(variant in ("r14", "r15", "r18", "r19", "r20", "r21"))
    val_cache = {}                       # R18 §29 same-runtime replay memo (pure)
    from .rolling_grpo import Graph  # noqa: F401
    cycles = int(cycles if cycles is not None else
                 {"A": C.TO1_R13_TRAINING_CYCLES_A,
                  "B": C.TO1_R13_TRAINING_CYCLES_B,
                  "C": C.TO1_R13_TRAINING_CYCLES_C}.get(stage, C.TO1_R13_TRAINING_CYCLES_C))
    k = int(k if k is not None else C.TO1_R13_K)
    horizon = int(horizon if horizon is not None else C.TO1_R13_HORIZON)
    gpb = int(graphs_per_batch if graphs_per_batch is not None else C.TO1_R13_GRAPHS_PER_BATCH)
    max_depth = int(C.TO1_R13_MAX_DEPTH)
    workers = int(workers)
    rng = random.Random(seed)

    collapse_ratio = float(C.TO1_R12_COLLAPSE_TRAIN_RATIO)
    r6_floor = float(collapse_floor) if collapse_floor is not None else None

    model_b5, single_head, direct_head = env["model_b5"], env["single_head"], env["direct_head"]
    executor = env["executor"]
    step0_cache = env["cache"]

    # §37-41 graph draw: seeded round-robin over DISTINCT instances (see _r13_cycle_picks)
    if per_src is None:
        per_src = {"bench": 2, "real": 1, "syn": 1} if not quick else {"bench": 2}
    cycle_handle = _r13_cycle_picks(
        schedule_specs, cycles, per_src, seed=int(seed), stage=str(stage),
        unified=bool(unified), per_cycle=per_cycle)

    history = []
    run_started = time.time()
    print(f"{log_prefix} progress [>-----------------------------] 0/{cycles} "
          "(0.0%) avg=calculating ETA=calculating", flush=True)
    best = {"cycle": -1, "score": -1e18, "policy": jpol.snapshot(),
            "train": 0, "real_held": 0, "syn_held": 0, "bench_held": 0}
    collapsed = False
    collapse_reason = None
    stage_a_base = None
    stage_a_gates = []
    stage_grad_steps = 0
    stage_prob_move = 0.0
    non_dry_stats = []
    for c in range(cycles):
        # T2-D controlled residual activation.  Zero-init still makes the first
        # forward exactly equal to the parent; alpha then warms linearly.
        for actor in (jpol.m2, jpol.m3):
            if hasattr(actor, "set_progress"):
                actor.set_progress(c + 1, cycles)
        t_start = time.time()
        active = cycle_handle[c]
        # fresh episodes every cycle: a graph reused across cycles must be pristine
        # when ITS cycle runs (build-time resets were undone by earlier-cycle advances)
        for _g in active:
            _g.reset()
        depth_results = []
        all_groups = []
        cycle_kl_m3, cycle_kl_m2 = [], []
        for depth in range(max_depth):
            act = [g for g in active if not g.done]
            if not act:
                break
            t_collect = time.time()
            depth_seed = seed + c * 1000 + depth * 100
            if use_r14 and depth_parallel and workers > 1:
                groups = collect_depth_groups_rollouts_r14(
                    jpol, scorer, executor, model_b5, single_head, direct_head,
                    act, k=k, horizon=horizon, seed=depth_seed, workers=workers,
                    step0_cache=step0_cache, mp_ctx=mp_ctx,
                    action_space=action_space, val_cache=val_cache,
                    critical_gate=worker_critical_gate,
                    torch_threads=worker_torch_threads,
                    rollout_pool=rollout_pool)
            else:
                groups = []
                for g in act:
                    grp = (collect_full_group_rollouts_r14 if use_r14
                           else collect_full_group_rollouts_r13)(
                        jpol, scorer, executor, model_b5, single_head, direct_head,
                        g.problem, g.schedule, g.ms, g.iid, g.episode_id, g.progmem,
                        k=k, horizon=horizon, seed=depth_seed,
                        step_offset=g.gstep, workers=workers,
                        step0_cache=step0_cache, mp_ctx=mp_ctx,
                        **({"action_space": action_space, "val_cache": val_cache}
                           if use_r14 else {}))
                    groups.append(grp)
            collect_s = time.time() - t_collect
            all_groups.extend(groups)
            if on_groups is not None:
                on_groups(groups)
            t_update = time.time()
            upd = grpo_update_joint(jpol, groups, stage=stage, seeds=seed,
                                    log_prefix=log_prefix,
                                    credit="stagewise" if use_r14 else "shared",
                                    optimizer_persistent=optimizer_persistent)
            update_s = time.time() - t_update
            if upd["epochs"]:
                if upd["epochs"][-1]["kl_ref_m3"] is not None:
                    cycle_kl_m3.append(upd["epochs"][-1]["kl_ref_m3"])
                cycle_kl_m2.append(upd["epochs"][-1]["kl_m2"])
            runtime_depth = _depth_runtime_metrics(
                groups, collect_s=collect_s, workers=workers,
                active_graphs=len(act))
            depth_results.append({"depth": depth,
                                  "n_groups": len(groups),
                                  "n_informative_trajectories": upd["n_informative_trajectories"],
                                  "n_trajectories": upd["n_trajectories"],
                                  "mean_reward": float(np.mean(
                                      [g["mean_reward"] for g in groups])) if groups else 0.0,
                                  "collect_s": round(collect_s, 4),
                                  "update_s": round(update_s, 4),
                                  "runtime": runtime_depth,
                                  "update": upd})
            t_advance = time.time()
            out = advance_graphs_r13(jpol, scorer, executor, model_b5, single_head,
                                     direct_head, act, rng, log_prefix=log_prefix,
                                     gate_variant=("r14" if use_r14 else "r13"),
                                     action_space=action_space)
            depth_results[-1]["advance_s"] = round(time.time() - t_advance, 4)
            depth_results[-1]["advance"] = out
        ev = {}
        t_eval = time.time()
        if eval_root_builder is not None:
            ev = eval_root_builder(jpol, cycle=c)
        eval_s = time.time() - t_eval
        train_gain = float(ev.get("train", {}).get("total", 0.0))
        real_hd = float(ev.get("real_held", {}).get("total", 0.0))
        syn_hd = float(ev.get("syn_held", {}).get("total", 0.0))
        bench_hd = float(ev.get("bench_held", {}).get("total", 0.0))
        score = (best_score_fn(ev) if best_score_fn is not None
                 else train_gain + real_hd + syn_hd)
        st_stats = _m2_probe_stats(all_groups)
        prob_move = _m2_prob_move(jpol, all_groups)
        cyc_n_info = sum(1 for g in all_groups if g["informative"])
        if stage == "A":
            n_grad_steps = sum(1 for d in depth_results
                               for e in d["update"]["epochs"]
                               if abs(e.get("grad_norm", 0.0)) > 1e-9)
            stage_grad_steps = stage_grad_steps + n_grad_steps
            stage_prob_move = max(stage_prob_move, float(prob_move))
            # A DRY cycle (no informative group -> no GRPO update) is a batch-sampling
            # miss, not a mechanism signal: exclude it from the aggregate stats below
            # (recorded for the report with dry=True).  The Stage-A summary gates are
            # computed at the END of the stage over all non-dry cycles.
            cyc_dry = bool(cyc_n_info == 0)
            if not cyc_dry:
                if stage_a_base is None:
                    stage_a_base = st_stats    # pre-training reference = first non-dry
                non_dry_stats.append(st_stats)
            stage_a_gates.append({"cycle": c, "stats": st_stats, "prob_move": prob_move,
                                  "n_informative": cyc_n_info, "dry": cyc_dry})
        m_kl_m3 = float(np.mean(cycle_kl_m3)) if cycle_kl_m3 else 0.0
        m_kl_m2 = float(np.mean(cycle_kl_m2)) if cycle_kl_m2 else 0.0
        cyc_n_inf2 = sum(d.get("update", {}).get(
            "n_informative_trajectories_m2", 0) for d in depth_results)
        reward_metrics = _cycle_reward_metrics(all_groups)
        runtime_rows = [d.get("runtime", {}) for d in depth_results]
        cycle_cache_lookups = sum(int(r.get("cache_lookups", 0)) for r in runtime_rows)
        cycle_cache_hits = sum(int(r.get("cache_hits", 0)) for r in runtime_rows)
        slowest_cycle = sorted(
            [dict(row, depth=d["depth"])
             for d in depth_results
             for row in d.get("runtime", {}).get("slowest_graphs", [])],
            key=lambda row: row.get("wall_s", 0.0), reverse=True)[:5]
        rollout_runtime = {
            "depths": runtime_rows,
            "effective_worker_utilization_mean": float(np.mean([
                r.get("effective_worker_utilization", 0.0) for r in runtime_rows
            ])) if runtime_rows else 0.0,
            "straggler_ratio_max": max((float(r.get("straggler_ratio", 0.0))
                                        for r in runtime_rows), default=0.0),
            "cache_hit_rate": cycle_cache_hits / max(cycle_cache_lookups, 1),
            "slowest_graphs": slowest_cycle,
        }
        dumped = {"cycle": c, "train": train_gain, "real_held": real_hd,
                  "syn_held": syn_hd, "bench_held": bench_hd, "score": score,
                  "n_groups": len(all_groups),
                  "n_informative": cyc_n_info,
                  "n_informative_trajectories_m2": cyc_n_inf2,
                  "informative_ratio": (cyc_n_info / max(len(all_groups), 1)),
                  "reward": reward_metrics,
                  "rollout_runtime": rollout_runtime,
                  "kl_ref_m3": m_kl_m3, "kl_m2": m_kl_m2,
                  "m2_stats": st_stats, "prob_move": prob_move,
                  "depth_results": depth_results,
                  "phase_s": {
                      "collect": round(sum(d.get("collect_s", 0.0)
                                           for d in depth_results), 4),
                      "update": round(sum(d.get("update_s", 0.0)
                                          for d in depth_results), 4),
                      "advance": round(sum(d.get("advance_s", 0.0)
                                           for d in depth_results), 4),
                      "eval": round(eval_s, 4),
                  },
                  "sec": round(time.time() - t_start, 2)}
        if tensorboard_writer is not None:
            last_epochs = [d["update"]["epochs"][-1] for d in depth_results
                           if d.get("update", {}).get("epochs")]
            def _avg(key):
                return float(np.mean([e.get(key, 0.0) for e in last_epochs])) if last_epochs else 0.0
            held_gain = float(np.mean([real_hd, syn_hd, bench_hd]))
            rt_depths = [d.get("runtime", {}) for d in depth_results]
            rt_weight = sum(max(int(r.get("active_graphs", 0)), 1) for r in rt_depths)

            def _rt_weighted(key):
                return (sum(float(r.get(key, 0.0)) * max(int(r.get("active_graphs", 0)), 1)
                            for r in rt_depths) / max(rt_weight, 1))

            cache_lookups = sum(int(r.get("cache_lookups", 0)) for r in rt_depths)
            cache_hits = sum(int(r.get("cache_hits", 0)) for r in rt_depths)
            tags = {
                "m2/grad_norm": _avg("grad_norm_m2"), "m2/kl": m_kl_m2,
                "m2/probe_positive_rate": st_stats["positive_probe_rate"],
                "m2/root_set_size": st_stats["probed_root_count"],
                "m2/unique_roots": float(len({
                    rec.get("state_hash") for g in all_groups for tr in g.get("trajs", [])
                    for rec in tr.get("steps", []) if rec.get("m2_rec")})),
                "m2/probes_per_positive_root": (st_stats["probed_root_count"] /
                    max(st_stats["tier_A"], 1.0)),
                "m3/grad_norm": _avg("grad_norm_m3"), "m3/kl": m_kl_m3,
                "m3/positive_trajectory_rate": (cyc_n_info / max(len(all_groups), 1)),
                "train/greedy_gain": train_gain, "held/greedy_gain": held_gain,
                "reward/mean": reward_metrics["mean"],
                "reward/median": reward_metrics["median"],
                "reward/std": reward_metrics["std"],
                "reward/min": reward_metrics["min"],
                "reward/max": reward_metrics["max"],
                "reward/positive_rate": reward_metrics["positive_rate"],
                "reward/negative_rate": reward_metrics["negative_rate"],
                "reward/zero_rate": reward_metrics["zero_rate"],
                "reward/mean_best_per_graph": reward_metrics["mean_best_per_graph"],
                "reward/informative_group_rate": reward_metrics["informative_group_rate"],
                "runtime/active_graphs_mean": (float(np.mean([
                    r.get("active_graphs", 0) for r in rt_depths]))
                    if rt_depths else 0.0),
                "runtime/submitted_shards_mean": (float(np.mean([
                    r.get("submitted_shards", 0) for r in rt_depths]))
                    if rt_depths else 0.0),
                "runtime/mean_graph_seconds": _rt_weighted("mean_graph_wall_s"),
                "runtime/max_graph_seconds": max(
                    (float(r.get("max_graph_wall_s", 0.0)) for r in rt_depths),
                    default=0.0),
                "runtime/straggler_ratio": max(
                    (float(r.get("straggler_ratio", 0.0)) for r in rt_depths),
                    default=0.0),
                "runtime/effective_worker_utilization": _rt_weighted(
                    "effective_worker_utilization"),
                "runtime/validation_seconds_aggregate": sum(
                    float(r.get("validation_seconds_aggregate", 0.0))
                    for r in rt_depths),
                "runtime/cache_hit_rate": cache_hits / max(cache_lookups, 1),
                "runtime/cache_entries_added": sum(
                    int(r.get("cache_entries_added", 0)) for r in rt_depths),
                "runtime/cache_entries_max_observed": max(
                    (int(r.get("cache_entries_max_observed", 0)) for r in rt_depths),
                    default=0),
            }
            for key in ("analyze", "execute", "mem", "policy", "m2gate"):
                tags[f"runtime/cpu_{key}_seconds"] = sum(
                    float(r.get("profile_cpu_s", {}).get(key, 0.0))
                    for r in rt_depths)
            for d in depth_results:
                tags[f"runtime/active_graphs_depth_{d['depth']}"] = float(
                    d.get("runtime", {}).get("active_graphs", 0))
                tags[f"runtime/shards_depth_{d['depth']}"] = float(
                    d.get("runtime", {}).get("submitted_shards", 0))
            tags.update(_t2d_actor_metrics(jpol, all_groups))
            completed = c + 1
            mean_cycle_s = ((time.time() - run_started) / completed)
            tags.update({"runtime/cycle_seconds": dumped["sec"],
                         "runtime/progress_percent": 100.0 * completed / cycles,
                         "runtime/eta_hours": mean_cycle_s * (cycles - completed) / 3600.0})
            for tag, value in tags.items():
                tensorboard_writer.add_scalar(tag, float(value), c)
            tensorboard_writer.flush()
        if r6_floor is not None and train_gain < collapse_ratio * r6_floor:
            collapsed = True
            collapse_reason = "train_gain"
            dumped["collapse"] = {"flag": True, "reason": collapse_reason}
        if score > best["score"] and not collapsed:
            best = {"cycle": c, "score": score, "policy": jpol.snapshot(),
                    "train": train_gain, "real_held": real_hd, "syn_held": syn_hd,
                    "bench_held": bench_hd}
        history.append(dumped)
        print(f"{log_prefix} [{stage}] cycle {c}: TRAIN={train_gain:.0f} "
              f"real_hd={real_hd:.0f} syn_hd={syn_hd:.0f} m2pos_rate="
              f"{st_stats['positive_probe_rate']:.3f} tierA={st_stats['tier_A']:.1f} "
              f"coverage={st_stats['coverage_ratio']:.3f} prob_move={prob_move:.4f} "
              f"kl_m2={m_kl_m2:.4f} kl_m3={m_kl_m3:.4f} sec={dumped['sec']}s "
              f"reward_mean={reward_metrics['mean']:.3f} "
              f"reward_max={reward_metrics['max']:.1f} "
              f"reward_pos={reward_metrics['positive_rate']:.3f} "
              f"rollout=<util={rollout_runtime['effective_worker_utilization_mean']:.2f} "
              f"straggler={rollout_runtime['straggler_ratio_max']:.2f} "
              f"cache={rollout_runtime['cache_hit_rate']:.2f} "
              f"slow={[(r.get('iid'), round(r.get('wall_s', 0.0), 1), r.get('depth')) for r in slowest_cycle[:3]]}> "
              f"phase={dumped['phase_s']}", flush=True)
        completed = c + 1
        elapsed = time.time() - run_started
        mean_cycle_s = elapsed / completed
        eta_s = mean_cycle_s * max(cycles - completed, 0)
        width = 30
        filled = min(width, int(width * completed / max(cycles, 1)))
        bar = ("=" * filled + (">" if filled < width else "")).ljust(width, "-")
        eta_h, eta_rem = divmod(int(eta_s), 3600)
        eta_m, eta_sec = divmod(eta_rem, 60)
        finish_at = time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(time.time() + eta_s))
        print(f"{log_prefix} progress [{bar}] {completed}/{cycles} "
              f"({100.0 * completed / max(cycles, 1):.1f}%) "
              f"avg={mean_cycle_s:.1f}s ETA={eta_h:02d}:{eta_m:02d}:{eta_sec:02d} "
              f"finish~{finish_at}", flush=True)
        if collapsed:
            break
    jpol.load_snapshot(best["policy"])
    stage_gates = _stage_a_gates_summary(
        stage_a_base if stage_a_base is not None else {},
        non_dry_stats, stage_grad_steps, stage_prob_move) if stage == "A" else {}
    return {"history": history, "best": best, "collapsed": collapsed,
            "collapse_reason": collapse_reason, "cycles_run": len(history),
            "stage_a_base": stage_a_base, "stage_a_gates": stage_a_gates,
            "stage_gates": stage_gates,
            "stage_grad_steps": stage_grad_steps, "stage_prob_move": stage_prob_move}


# ---------------------------------------------------------------------------
# JointAgenticPolicy -- one object, two trainable systems  (§9-11,19-20,28)
# ---------------------------------------------------------------------------
class JointAgenticPolicy:
    """wraps M3RollingGRPOPolicy (proposal selector, parent m3_rolling_grpo_v1.pt
    warm start, alpha_M3=0.5 kept) + M2RootPolicyAdapter (root search residual on
    frozen B5 attribution).  Stage A/B/C decide which parameters live in the GRPO
    optimizer; both share the SAME per-trajectory terminal advantage."""

    def __init__(self, r6_selector, alpha_prop=None, alpha_stop=None, alpha_m2=None,
                 m3_builder=None, m2_builder=None):
        # R18: m3_builder overrides the default M3RollingGRPOPolicy so the joint loop
        # can carry M3ProposalEvidencePolicy (frozen R6 + zero-init evidence residual)
        # while M2/M3 optics (alpha attr, snapshot, params_for_stage) stay identical.
        if m3_builder is not None:
            self.m3 = m3_builder(r6_selector)
        else:
            self.m3 = M3RollingGRPOPolicy(r6_selector,
                                          alpha_prop=TO1_R13_ALPHA_PROP
                                          if alpha_prop is None else alpha_prop,
                                          alpha_stop=TO1_R13_ALPHA_STOP
                                          if alpha_stop is None else alpha_stop)
        if m2_builder is not None:
            self.m2 = m2_builder()
        else:
            self.m2 = M2RootPolicyAdapter(TO1_R13_M2_FEAT_DIM,
                                          alpha_m2=TO1_R13_ALPHA_M2
                                          if alpha_m2 is None else alpha_m2)

    @property
    def m3_params(self):
        return [p for p in self.m3.parameters() if p.requires_grad]

    @property
    def m2_params(self):
        return [p for p in self.m2.parameters() if p.requires_grad]

    def freeze_m3(self):
        for p in self.m3.parameters():
            p.requires_grad_(False)

    def unfreeze_m3(self):
        for p in self.m3.parameters():
            p.requires_grad_()

    def freeze_m2(self):
        for p in self.m2.parameters():
            p.requires_grad_(False)

    def unfreeze_m2(self):
        for p in self.m2.parameters():
            p.requires_grad_()

    def params_for_stage(self, stage):
        if stage == "A":
            self.freeze_m3()
            self.unfreeze_m2()
        elif stage == "B":
            self.freeze_m2()
            self.unfreeze_m3()
        else:
            self.unfreeze_m2()
            self.unfreeze_m3()

    def snapshot(self):
        return {"m3": self.m3.snapshot(), "m2": self.m2.snapshot()}

    def load_snapshot(self, snap):
        self.m3.load_snapshot(snap["m3"])
        self.m2.load_snapshot(snap["m2"])


# ---------------------------------------------------------------------------
# deterministic greedy agentic evaluator -- CANONICAL same-ruler closed loop
# ---------------------------------------------------------------------------
def agentic_greedy_step(jpol, scorer, executor, model_b5, single_head, direct_head,
                        problem, schedule, iid, episode_id, progmem, root_ms,
                        step_offset=0, gate_mem=False, gate_variant="r13",
                        action_space="full"):
    """Deterministic greedy action under the §0 gated pool (adapter active).  Same
    body as R12 `greedy_step_unified` with the M2 gate inserted; never mutates inputs.
    gate_variant "r14" uses the adaptive probe + q2 gate (`_m2_gate_step_r14`).
    action_space "shortlist" (R15) replaces the wide_pool stage with the
    coverage-preserving shortlist (§5-9) so eval shares the R15 M3 action set."""
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
    rng = random.Random(0)
    _gatefn = _m2_gate_step_r14 if gate_variant == "r14" else _m2_gate_step
    gate = _gatefn(ast, metas, prop_feats, jpol.m2, executor, progmem, iid,
                   episode_id, step_offset, sf, rng)
    if not gate["gated_metas"]:
        return {"action": "stop", "reason": "no_pool_m2", "state_hash": h,
                "m2_diag": gate["diag"]}
    rolex = _rollex_of(ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(progmem.features(iid, episode_id, step_offset, sf, queries),
                       dtype=torch.float32)
    if gate_mem:
        mem = mem * float(progmem.retrieval_gate(iid, episode_id, step_offset, sf))
    evid = None
    _vs_ready = False
    if action_space == "shortlist":
        shortlist_idx, _sl_info = build_shortlist_r15(
            ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t, jpol, scorer, mem,
            gate["m2_rec"]["tier_a"], gate["m2_rec"]["tier_b"], rng,
            cap=C.TO1_R15_SHORTLIST_CAP, k_global=C.TO1_R15_K_GLOBAL,
            rolex=rolex)
        if not shortlist_idx:
            out = {"action": "stop", "reason": "no_pool_shortlist", "state_hash": h,
                   "m2_diag": gate["diag"]}
            return out
        pool = shortlist_idx
    elif action_space == "validated":
        # R18 §12-13: eval shares the canonical validated action set.  The M3
        # evidence observation (evid) is built once here and passed to logits.
        vs = build_validated_action_set_r18(
            ast, gate, rolex, scorer, executor, progmem, iid, episode_id,
            step_offset, sf, h, ms_cur, mem)
        if not vs["pool"]:
            out = {"action": "stop", "reason": "no_validated_pool", "state_hash": h,
                   "m2_diag": gate["diag"]}
            return out
        pool = vs["pool"]
        F_pool = vs["F_pool"]
        pool_stats = vs["pool_stats"]
        evid = vs["evid"]
        _vs_ready = True
    elif action_space == "multistep":
        # R19 §6-24: eval shares the canonical H-step-validated action set, same
        # as collect.  G1/GH are observation-only evidence; M3 still selects.
        vs = build_multistep_action_set_r19(
            ast, gate, rolex, scorer, executor, progmem, iid, episode_id,
            step_offset, sf, h, ms_cur, mem,
            cont_m2=getattr(jpol, "cont_m2", None),
            cont_m3=getattr(jpol, "cont_m3", None),
            acache=cache, rc=None, hcache=None)
        if not vs["pool"]:
            out = {"action": "stop", "reason": "no_multistep_pool", "state_hash": h,
                   "m2_diag": gate["diag"]}
            return out
        pool = vs["pool"]
        F_pool = vs["F_pool"]
        pool_stats = vs["pool_stats"]
        evid = vs["evid"]
        _vs_ready = True
    elif action_space == "lexicographic":
        # R20 §4-12: eval shares the state-level lexicographic fallback set.
        vs = build_lexicographic_action_set_r20(
            ast, gate, rolex, scorer, executor, progmem, iid, episode_id,
            step_offset, sf, h, ms_cur, mem,
            cont_m2=getattr(jpol, "cont_m2", None),
            cont_m3=getattr(jpol, "cont_m3", None),
            acache=cache, rc=None, hcache=None)
        if not vs["pool"]:
            out = {"action": "stop", "reason": "no_lexicographic_pool", "state_hash": h,
                   "m2_diag": gate["diag"]}
            return out
        pool = vs["pool"]
        F_pool = vs["F_pool"]
        pool_stats = vs["pool_stats"]
        evid = vs["evid"]
        _vs_ready = True
    else:
        logit_pos, rank = _scores(scorer, rolex, mem)
        pool, pinfo = wide_pool(rolex, logit_pos, rank)
    out = {"action": "stop", "reason": "no_pool_wide", "state_hash": h,
           "m2_diag": gate["diag"]}
    if not pool:
        return out
    if not _vs_ready:
        F_pool = _rerank_feats_all(scorer, rolex, mem)[pool]
        pool_stats = _pool_stats_from(F_pool)
    with torch.no_grad():
        logits = selector_action_logits(jpol.m3, F_pool, sf_t, pool_stats, evid=evid)
    out.update({"m2_diag": gate["diag"], "logits": logits.detach().float()})
    sel = int(logits.argmax().item())
    out["M"] = len(pool)
    out["best_pool_score"] = float(logits[:len(pool)].max())
    out["stop_score"] = float(logits[-1])
    if sel == len(pool):
        out.update({"action": "stop", "reason": "policy_stop", "state_hash": h})
        return out
    meta = gate["gated_metas"][pool[sel]]
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
                "sig": sig, "kind": kind,
                "meta_type": rolex["type"][pool[sel]],
                "meta_role": rolex["role"][pool[sel]],
                "src": rolex["src"][pool[sel]], "tgt": rolex["tgt"][pool[sel]],
                "improvement": float(res["improvement"]), "successor": res["schedule"]})
    return out


def agentic_parity_rollout(env, scorer, jpol, rf, use_mem=True, horizon=None,
                           gate_mem=False, gate_variant="r13", action_space="full",
                           sample=None, allow_policy_stop=True,
                           stop_on_nonpositive=True, feasible_fallback=False):
    """Deterministic greedy closed loop under the SAME canonical ruler as R12
    `unified_parity_rollout` (STOP-on-negative, same Runtime/Memory/pool-norm/horizon)
    with the §0 M2 gate inserted -- the C2/C3/C4/C5 evaluator.  m2_mode="none"
    (baselines C0/C1) reuses `unified_parity_rollout` untouched so the 369 anchor
    reproduces bit-identically.  gate_variant="r14" runs the ADAPTIVE probe + q2 gate
    so train and eval share the identical §0 permanent structure."""
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
    last_step_gain = 0.0
    non_improving_streak = 0
    n_acted = 0
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
        rng = (sample["rng"] if action_space == "policy_sampled" and sample is not None
               else random.Random(_traj_seed(0, iid, episode_id, h, 0)))
        if action_space == "policy_sampled":
            gate = _m2_policy_sample_gate(
                ast, metas, prop_feats, jpol.m2, progmem, iid, episode_id,
                t, sf, rng, greedy=(sample is None))
        else:
            _gate = _m2_gate_step_r14 if gate_variant == "r14" else _m2_gate_step
            gate = _gate(ast, metas, prop_feats, jpol.m2, executor, progmem,
                         iid, episode_id, t, sf, rng)
        if not gate["gated_metas"]:
            act_usage["stop_by_selector"] += 1
            steps.append({"t": t, "n_prop": n_prop, "n_wide": 0, "top_sig": None,
                          "kind": None, "improvement": None,
                          "m2_diag": gate["diag"], "stop_reason": "no_pool_m2"})
            break
        rolex = _rollex_of(ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t)
        queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
                   for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                             rolex["src"], rolex["tgt"])]
        mem = (torch.tensor(progmem.features(iid, episode_id, t, sf, queries),
                            dtype=torch.float32) if use_mem
               else torch.zeros(len(queries), MEM_FEAT_DIM, dtype=torch.float32))
        gmem = 1.0
        if gate_mem and use_mem:
            gmem = float(progmem.retrieval_gate(iid, episode_id, t, sf))
        mem_sel = mem * gmem if gate_mem else mem
        evid = None
        _vs_ready = False
        _pvdiag = None
        pair_mask = None
        traj_ctx = trajectory_context(
            step_index=t, horizon=horizon, root_makespan=s0_ms,
            current_makespan=ms_cur, last_step_gain=last_step_gain,
            action_count=n_acted, non_improving_streak=non_improving_streak)
        if action_space == "policy_sampled":
            F_all = _rerank_feats_all(scorer, rolex, mem_sel)
            pool, _pair_cap, _pool_info = _rl_hierarchical_action_pool(
                jpol, gate["gated_metas"], F_all, sf_t, traj_ctx,
                non_improving_streak, rng, greedy=(sample is None))
            if not pool:
                act_usage["stop_by_selector"] += 1
                steps.append({"t": t, "n_prop": n_prop, "n_wide": 0,
                              "top_sig": None, "kind": None,
                              "improvement": None, "m2_diag": gate["diag"],
                              "stop_reason": "no_policy_sampled_pool"})
                break
            F_pool = F_all[pool]
            pool_stats = _pool_stats_from(F_pool)
            _vs_ready = True
        elif action_space == "shortlist":
            shortlist_idx, _sl_info = build_shortlist_r15(
                ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t, jpol, scorer, mem,
                gate["m2_rec"]["tier_a"], gate["m2_rec"]["tier_b"], rng,
                cap=C.TO1_R15_SHORTLIST_CAP, k_global=C.TO1_R15_K_GLOBAL,
                rolex=rolex)
            if not shortlist_idx:
                act_usage["stop_by_selector"] += 1
                steps.append({"t": t, "n_prop": n_prop, "n_wide": 0, "top_sig": None,
                              "kind": None, "improvement": None,
                              "m2_diag": gate["diag"], "pv_diag": _pvdiag,
                              "stop_reason": "no_pool_shortlist"})
                break
            pool = shortlist_idx
        elif action_space == "validated":
            # R18 §12-13 canonical eval action set: all validated Tier-A + capped
            # Tier-B + STOP, no cap32, no extra Top-K.  evid is the observation
            # M3 evidence residual is allowed to see (§17-20); nothing more.
            vs = build_validated_action_set_r18(
                ast, gate, rolex, scorer, executor, progmem, iid, episode_id,
                t, sf, h, ms_cur, mem_sel)
            if not vs["pool"]:
                act_usage["stop_by_selector"] += 1
                steps.append({"t": t, "n_prop": n_prop, "n_wide": 0, "top_sig": None,
                              "kind": None, "improvement": None,
                              "m2_diag": gate["diag"], "stop_reason": "no_validated_pool"})
                break
            pool = vs["pool"]
            F_pool = vs["F_pool"]
            pool_stats = vs["pool_stats"]
            evid = vs["evid"]
            _pvdiag = vs["pval"]["diag"]
            _vs_ready = True
        elif action_space == "multistep":
            # R19 §6-24 canonical eval action set (G1 -> GH -> Memory tiers),
            # identical to collect.  `acache` = env AnalyzeCache for branch pools.
            vs = build_multistep_action_set_r19(
                ast, gate, rolex, scorer, executor, progmem, iid, episode_id,
                t, sf, h, ms_cur, mem_sel,
                cont_m2=getattr(jpol, "cont_m2", None),
                cont_m3=getattr(jpol, "cont_m3", None),
                acache=cache, rc=None, hcache=None)
            if not vs["pool"]:
                act_usage["stop_by_selector"] += 1
                steps.append({"t": t, "n_prop": n_prop, "n_wide": 0, "top_sig": None,
                              "kind": None, "improvement": None,
                              "m2_diag": gate["diag"], "stop_reason": "no_multistep_pool"})
                break
            pool = vs["pool"]
            F_pool = vs["F_pool"]
            pool_stats = vs["pool_stats"]
            evid = vs["evid"]
            _pvdiag = vs["pval"]["diag"]
            _vs_ready = True
        elif action_space == "lexicographic":
            # R20 §4-12: canonical eval action set = state-level lexicographic
            # fallback, identical to collect.
            vs = build_lexicographic_action_set_r20(
                ast, gate, rolex, scorer, executor, progmem, iid, episode_id,
                t, sf, h, ms_cur, mem_sel,
                cont_m2=getattr(jpol, "cont_m2", None),
                cont_m3=getattr(jpol, "cont_m3", None),
                acache=cache, rc=None, hcache=None,
                feasible_fallback=bool(feasible_fallback))
            if not vs["pool"]:
                act_usage["stop_by_selector"] += 1
                steps.append({"t": t, "n_prop": n_prop, "n_wide": 0, "top_sig": None,
                              "kind": None, "improvement": None,
                              "m2_diag": gate["diag"],
                              "stop_reason": "no_lexicographic_pool"})
                break
            pool = vs["pool"]
            F_pool = vs["F_pool"]
            pool_stats = vs["pool_stats"]
            evid = vs["evid"]
            _pvdiag = vs["pval"]["diag"]
            _vs_ready = True
        else:
            logit_pos, rank = _scores(scorer, rolex, mem)
            pool, _info = wide_pool(rolex, logit_pos, rank)
            if not pool:
                act_usage["stop_by_selector"] += 1
                steps.append({"t": t, "n_prop": n_prop, "n_wide": 0, "top_sig": None,
                              "kind": None, "improvement": None,
                              "m2_diag": gate["diag"], "pv_diag": _pvdiag,
                              "stop_reason": "no_pool_wide"})
                break
        if not _vs_ready:
            F_pool = _rerank_feats_all(scorer, rolex, mem_sel)[pool]
            pool_stats = _pool_stats_from(F_pool)
        if action_space == "policy_sampled":
            pair_mask = torch.tensor([
                gate["gated_metas"][k].get("kind") == "pair" for k in pool
            ], dtype=torch.bool)
        with torch.no_grad():
            logits = selector_action_logits(
                jpol.m3, F_pool, sf_t, pool_stats, evid=evid, traj_ctx=traj_ctx,
                pair_mask=pair_mask)
        selectable_logits = logits if bool(allow_policy_stop) else logits[:len(pool)]
        if sample is None:
            # Greedy reference path (§12 / R12-R21 canonical argmax ruler).
            sel = int(selectable_logits.argmax().item())
        else:
            # T2-A stochastic branch: reuse the SAME mixture sampler as GRPO collect
            # (temperature + uniform mass), never re-invented.  rng is per-branch
            # (caller-owned) so each sampled trajectory is a deterministic replay
            # given its seed; STOP (index len(pool)) is a legal draw.
            sel = int(mixture_sample(
                selectable_logits, sample["T"], sample["eps"], sample["rng"]))
        if sel == len(pool):
            act_usage["stop_by_selector"] += 1
            steps.append({"t": t, "n_prop": n_prop, "n_wide": len(pool),
                          "top_sig": "STOP", "kind": None, "improvement": None,
                          "m2_diag": gate["diag"], "pv_diag": _pvdiag,
                          "stop_reason": "policy_stop"})
            break
        a = pool[sel]
        edits, kind = _edits_for(ast, gate["gated_metas"][a])
        res = _execute_step(executor, problem, schedule, edits, ms_cur, h)
        sig = proposal_identity(ast, gate["gated_metas"][a])[2]
        steps.append({"t": t, "n_prop": n_prop, "n_wide": len(pool),
                      "top_sig": sig, "kind": kind,
                      "improvement": (None if res is None else float(res["improvement"])),
                      "m2_diag": gate["diag"], "pv_diag": _pvdiag})
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
        if res is None or (bool(stop_on_nonpositive) and res["improvement"] <= 0):
            act_usage["stop_neg"] += 1
            break
        act_usage[kind] += 1
        n_acted += 1
        last_step_gain = float(res["improvement"])
        non_improving_streak = (0 if float(res["improvement"]) > 0.0
                                else non_improving_streak + 1)
        schedule = res["schedule"]
        ms_cur = int(schedule.makespan)
        nh = schedule_hash(schedule)
        if nh in visited:
            break
        visited.add(nh)
    return int(schedule0.makespan) - ms_cur, act_usage, steps


def agentic_closed_loop(env, scorer, jpol, roots, use_mem=True, horizon=None,
                        gate_mem=False, gate_variant="r13", action_space="full"):
    """Same contract as `unified_parity_closed_loop` but under the §0 gated pool.
    Returns (_b5_summary-shaped gains, steps_by_iid)."""
    gains_by_iid = {}
    steps_by_iid = {}
    for rf in roots:
        iid = rf["iid"]
        gain, usage, steps = agentic_parity_rollout(
            env, scorer, jpol, rf, use_mem=use_mem, horizon=horizon, gate_mem=gate_mem,
            gate_variant=gate_variant, action_space=action_space)
        gains_by_iid[iid] = gain
        steps_by_iid[iid] = {"usage": usage, "steps": steps}
    from .rollout import _b5_summary
    return _b5_summary(gains_by_iid), steps_by_iid


def parity_eval(env, scorer, selector_or_jpol, roots, m2_mode="none",
                use_mem=True, horizon=None, gate_mem=False, gate_variant="r13",
                action_space="full"):
    """One dispatch for the CANONICAL same-ruler evaluator across P0..P3:
    m2_mode="none"  -> R12 `unified_parity_closed_loop` (existing M2, no gate) so the
                       anchor rows reproduce the R12 369 ruler bit-identically.
    m2_mode="adapter" -> `agentic_closed_loop` (§0 gate, adapter active).  For R14 the
                       gate is the ADAPTIVE one (gate_variant="r14") so train + eval
                       share the identical permanent structure (§0).  R15 action_space
                       "shortlist" (Q1/Q3) restricts the M3 action set to the
                       coverage-preserving shortlist."""

    if m2_mode == "none":
        return unified_parity_closed_loop(env, scorer, selector_or_jpol, roots,
                                          use_mem=use_mem, horizon=horizon,
                                          gate_mem=gate_mem)
    return agentic_closed_loop(env, scorer, selector_or_jpol, roots, use_mem=use_mem,
                               horizon=horizon, gate_mem=gate_mem,
                               gate_variant=gate_variant, action_space=action_space)


# ---------------------------------------------------------------------------
# T2-A -- MULTI-PATH HELD TRAJECTORY EVALUATION (pure metric/verdict helpers)
# Evaluation-only: these consume `agentic_parity_rollout` outputs to measure
# whether N sampled trajectories expose latent positive paths greedy misses.
# No training / no oracle / no state mutation.
# ---------------------------------------------------------------------------
def t2a_branch_seed(iid, episode_id, branch_idx):
    """§29 deterministic-replay seed: stable across processes (md5, not PYTHONHASHSEED)."""
    import hashlib
    key = f"t2a|{iid}|{int(episode_id)}|{int(branch_idx)}|{int(C.TO1_T2A_SEED_BASE)}"
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def t2a_traj_sig(steps):
    """Lightweight trajectory signature for diversity counting (distinct explored paths)."""
    return tuple((int(s.get("t", 0)), s.get("top_sig"), s.get("kind")) for s in steps)


def t2a_compact_steps(steps):
    """Drop the heavy m2_diag/pv_diag nested dicts for a JSON-safe per-instance trace."""
    return [{"t": s.get("t"), "top_sig": s.get("top_sig"), "kind": s.get("kind"),
             "improvement": s.get("improvement"), "stop_reason": s.get("stop_reason")}
            for s in steps]


def t2a_classify_failure(greedy, samples, sigs, step_counts):
    """§35 failure decomposition (mutually-exclusive, priority order)."""
    best_of_N = int(max(samples)) if samples else int(greedy)
    best_overall = int(max([greedy] + samples))
    n = len(samples)
    all_zero = all(c == 0 for c in step_counts)
    all_le1 = all(c <= 1 for c in step_counts)
    n_distinct = len(set(sigs))
    if all_zero:
        return "CANDIDATE_GENERATION_MISS"
    if greedy <= 0 and best_of_N > 0:
        return "M3_SELECTION_DISTRIBUTION_MISS"
    if n > 1 and n_distinct == 1:
        return "POLICY_ENTROPY_COLLAPSE"
    if all_le1:
        return "EARLY_STOP_COLLAPSE"
    if best_overall <= 0:
        return "NO_POSITIVE_PATH_FOUND"
    return "POSITIVE_PATH_PRESENT"


def t2a_split_summary(rows):
    """Aggregate per-instance rows into split-level metric block (§14-24)."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "greedy_total": 0, "bestN_total": 0, "bestoverall_total": 0,
                "mean_total": 0.0, "median_total": 0.0, "success_at_N": 0.0,
                "success_overall": 0.0, "greedy_pos_rate": 0.0,
                "positive_traj_rate": 0.0, "latent_n": 0, "latent_denom": 0,
                "latent_rate": 0.0, "bestofN_gain": 0, "sampled_best_gap": 0.0,
                "mean_distinct_traj": 0.0, "mean_distinct_outcomes": 0.0,
                "failure": {}}
    N = int(C.TO1_T2A_N_SAMPLED)
    greedy_total = int(sum(r["greedy"] for r in rows))
    bestN_total = int(sum(r["best_of_N"] for r in rows))
    bestoverall_total = int(sum(r["best_overall"] for r in rows))
    mean_total = float(sum(r["mean"] for r in rows))
    median_total = float(sum(r["median"] for r in rows))
    n_greedy_pos = int(sum(1 for r in rows if r["greedy"] > 0))
    n_bestN_pos = int(sum(1 for r in rows if r["best_of_N"] > 0))
    n_bestoverall_pos = int(sum(1 for r in rows if r["best_overall"] > 0))
    n_pos_traj = int(sum(r["n_pos_traj"] for r in rows))
    latent_rows = [r for r in rows if r["latent"]]
    greedy_le0 = [r for r in rows if r["greedy"] <= 0]
    latent_n = len(latent_rows)
    latent_denom = len(greedy_le0)
    return {
        "n": n,
        "greedy_total": greedy_total, "bestN_total": bestN_total,
        "bestoverall_total": bestoverall_total,
        "mean_total": mean_total, "median_total": median_total,
        "success_at_N": (n_bestN_pos / max(n, 1)),
        "success_overall": (n_bestoverall_pos / max(n, 1)),
        "greedy_pos_rate": (n_greedy_pos / max(n, 1)),
        "positive_traj_rate": (n_pos_traj / max(n * N, 1)),
        "latent_n": latent_n, "latent_denom": latent_denom,
        "latent_rate": (latent_n / max(latent_denom, 1)),
        "bestofN_gain": (bestN_total - greedy_total),
        "sampled_best_gap": (float(np.mean([r["best_of_N"] - r["greedy"] for r in rows]))
                             if n else 0.0),
        "mean_distinct_traj": (float(np.mean([r["n_distinct_traj"] for r in rows]))
                               if n else 0.0),
        "mean_distinct_outcomes": (float(np.mean([r["n_distinct_outcomes"] for r in rows]))
                                   if n else 0.0),
        "failure": dict(Counter(r["failure"] for r in rows)),
    }


def t2a_verdict(repro_ok, held, held_n, mp_ok=True):
    """§41-48 verdict ladder A-H.  GO_5070TI only on A (§49)."""
    if not repro_ok:
        return "G", "R6_ANCHOR_REPRO_FAILURE", \
               "R6 reproduction anchor failed -- runtime/wiring bug; abort before reading any result", False
    if not mp_ok:
        return "G", "MULTIPROCESS_INCONSISTENCY", "cloud profile not identical across workers", False
    if held_n == 0:
        return "F", "NO_HELD_DATA", \
               "no held instances available (quick mode without AUX); latent-signal question untestable", False
    latent_n = int(held.get("latent_n", 0))
    latent_rate = float(held.get("latent_rate", 0.0))
    bestofN_gain = float(held.get("bestofN_gain", 0.0))
    mean_total = float(held.get("mean_total", 0.0))
    greedy_total = float(held.get("greedy_total", 0.0))
    mean_distinct = float(held.get("mean_distinct_traj", 0.0))
    fail = held.get("failure", {})
    n = int(held.get("n", 0))
    collapse_frac = ((fail.get("POLICY_ENTROPY_COLLAPSE", 0)
                      + fail.get("EARLY_STOP_COLLAPSE", 0)) / max(n, 1))
    collapsed = bool(mean_distinct <= 1.05) or bool(collapse_frac > 0.5)
    if collapsed:
        return "E", "POLICY_ENTROPY_OR_EARLY_STOP_COLLAPSE", \
               (f"sampling explores nothing on held: mean_distinct_traj={mean_distinct:.2f} "
                f"collapse_frac={collapse_frac:.2f} -- distinct problem from no-latent-signal"), False
    if latent_n >= 1:
        if latent_rate >= 0.2 and latent_n >= 2 and bestofN_gain > 0:
            return "A", "LATENT_POSITIVE_TRAJECTORIES_CONFIRMED", \
                   (f"held latent_n={latent_n} latent_rate={latent_rate:.3f} "
                    f"bestofN_gain={bestofN_gain:+.0f} -- multi-path exposes latent positive "
                    f"trajectories single greedy misses; direction holds"), True
        return "B", "LATENT_SIGNAL_PRESENT_BUT_WEAK", \
               (f"held latent_n={latent_n} latent_rate={latent_rate:.3f} "
                f"bestofN_gain={bestofN_gain:+.0f} -- signal present but below A threshold "
                f"(rate<0.2 or n<2 or gain<=0); worth larger N / better sampler"), False
    # latent_n == 0
    if mean_total < greedy_total:
        return "D", "SAMPLING_DEGRADES_ON_AVERAGE", \
               (f"held mean_total={mean_total:.0f} < greedy_total={greedy_total:.0f} and no "
                f"latent path -- sampling is net-negative on held"), False
    return "C", "NO_LATENT_SIGNAL_MULTI_PATH_INSUFFICIENT", \
           (f"held latent_n=0 over {max(held.get('latent_denom', 0), 1)} greedy<=0 instances -- "
            f"multi-path alone cannot rescue; structural bottleneck (candidate-pool ceiling / "
            f"multi-step credit) remains"), False


# ---------------------------------------------------------------------------
# T2-B -- MULTI-PATH JOINT AGENTIC GRPO diagnostics (pure, no training)
# ExtractionGap = 差值 (bestN_total - greedy_total) = t2a_split_summary["bestofN_gain"].
# best-of-N is a post-hoc diagnostic ONLY -- never reward/advantage/oracle.
# ---------------------------------------------------------------------------
def t2b_l2g_conversion(e0_rows, final_gains):
    """Latent-to-Greedy conversion: fraction of E0-latent roots (greedy<=0, bestN>0)
    whose FINAL greedy gain is now >0.  Pure.  e0_rows carry per-instance 'latent'
    + 'iid'; final_gains = {iid: final_greedy_gain}."""
    latent = [r for r in e0_rows if r.get("latent")]
    if not latent:
        return 0.0
    conv = sum(1 for r in latent if float(final_gains.get(r["iid"], 0.0)) > 0)
    return float(conv) / float(len(latent))


def t2b_useful_group_rate(groups):
    """Fraction of collected groups that are informative (terminal std > adv_eps)."""
    n = len(groups)
    return float(sum(1 for g in groups if g.get("informative")) / max(n, 1))


def t2b_all_same_rate(groups):
    """Fraction of groups where all K sibling terminal rewards are identical."""
    n = len(groups)
    return float(sum(1 for g in groups if len(set(g.get("rewards", []))) == 1)
                 / max(n, 1))


def t2b_first_action_credit(groups):
    """Front-loaded reward diagnostic: over positive terminal trajectories, what
    fraction of the terminal reward is locked in by the FIRST executed action
    (its validation_gain, §52/53 assert it equals the executed single-step gain)."""
    sum_first = 0.0
    sum_term = 0.0
    n_pos = 0
    for g in groups:
        for tr in g.get("trajs", []):
            rew = float(tr.get("reward", 0.0))
            if rew <= 0:
                continue
            n_pos += 1
            first = None
            for rec in tr.get("steps", []):
                if rec.get("validation_gain") is not None:
                    first = float(rec["validation_gain"])
                    break
            if first is not None:
                sum_first += first
            sum_term += rew
    return {"n_pos_traj": n_pos, "sum_first_gain": sum_first,
            "sum_terminal": sum_term,
            "credit": (sum_first / sum_term) if sum_term > 0 else 0.0}


def t2b_verdict(repro_ok, mp_ok, m5_ok, e0, final, held_n):
    """§A-H/I ladder + GO_5070TI on A only.  Primary success = 双闸 (both gates):
    gate-1 greedy 转正 (final.greedy_total > e0.greedy_total + margin) AND
    gate-2 gap 收窄 (final.bestofN_gain <= REDUCE * e0.bestofN_gain, 差值)."""
    if not repro_ok:
        return "G", "R6_ANCHOR_REPRO_FAILURE", \
               "R6 anchor failed -- runtime/wiring bug; abort before reading any result", False
    if not mp_ok:
        return "G", "MULTIPROCESS_INCONSISTENCY", \
               "cloud profile not identical across workers", False
    if not m5_ok:
        return "H", "RUNTIME_SEMANTICS_REGRESSION", \
               "hard-freeze violated (normal-M5/FDR/STOP legality changed)", False
    if held_n == 0:
        return "I", "NO_HELD_DATA", \
               "no held instances available; extraction-gap untestable", False
    e0_gap = float(e0.get("bestofN_gain", 0.0))
    if e0_gap <= 0:
        return "F", "NO_LATENT_SIGNAL_AT_E0", \
               "E0 best-of-N gain <= 0 -- no latent positive trajectory to extract", False
    mean_distinct = float(final.get("mean_distinct_traj", 0.0))
    if mean_distinct <= 1.05:
        return "E", "POLICY_ENTROPY_COLLAPSE", \
               "sampling explores nothing on held after training", False
    e0_greedy = float(e0.get("greedy_total", 0.0))
    fin_greedy = float(final.get("greedy_total", 0.0))
    margin = float(C.TO1_T2B_GREEDY_HELD_MARGIN)
    if fin_greedy < e0_greedy - margin:
        return "D", "TRAINING_DEGRADES_HELD", \
               (f"held greedy collapsed below E0: {e0_greedy:+.0f} -> {fin_greedy:+.0f}"), False
    fin_gap = float(final.get("bestofN_gain", 0.0))
    g1 = bool(fin_greedy > e0_greedy + margin)                                  # gate 1
    g2 = bool(fin_gap <= float(C.TO1_T2B_EXTRACTION_GAP_REDUCE) * e0_gap)       # gate 2
    if g1 and g2:
        return "A", "EXTRACTION_SUCCESS", \
               (f"greedy {e0_greedy:+.0f}->{fin_greedy:+.0f} gap "
                f"{e0_gap:+.0f}->{fin_gap:+.0f} -- latent positives greedy-extracted"), True
    if g1 or g2:
        return "B", "PARTIAL_EXTRACTION", \
               (f"one gate only: greedy_up={g1} gap_down={g2} "
                f"(greedy {e0_greedy:+.0f}->{fin_greedy:+.0f}, gap {e0_gap:+.0f}->{fin_gap:+.0f})"), False
    return "C", "NO_EXTRACTION", \
           (f"neither greedy turned positive nor gap narrowed "
            f"(greedy {e0_greedy:+.0f}->{fin_greedy:+.0f}, gap {e0_gap:+.0f}->{fin_gap:+.0f})"), False


# ---------------------------------------------------------------------------
# diagnostics -- per-state / per-cycle R13 metrics + failure decomposition  (§42-47)
# ---------------------------------------------------------------------------
def state_m2_diag(ast, metas, prop_feats, adapter, executor, pm, iid, episode_id,
                  step, sf, rng):
    """Cheap per-state M2 gate diagnostic (one probe pass; used by the eval traces
    and the §42/§46 metric families)."""
    gate = _m2_gate_step(ast, metas, prop_feats, adapter, executor, pm, iid,
                         episode_id, step, sf, rng)
    return {"diag": gate["diag"], "retained": gate["retained"],
            "probe": gate["m2_rec"]["probe"], "gated_metas": gate["gated_metas"]}


def _positive_proposals_in_pool(ast, gate, probed):
    """Tier-A roots guarantee their (probed-positive) single/pair proposals stay in the
    gated pool -- this is a pure sanity read from the gate's own probe evidence."""
    pos_root_ops = {op for op, p in probed.items() if p["g_probe"] > 0.0}
    keep = [m for m in gate["gated_metas"]]
    pool = ast["pool"]
    known_pos = 0
    for m in keep:
        if m["kind"] == "single":
            if pool[m["i"]]["e"].operation_id in pos_root_ops:
                known_pos += 1
        else:
            oa, ob = (pool[m["i"]]["e"].operation_id, pool[m["j"]]["e"].operation_id)
            if oa in pos_root_ops or ob in pos_root_ops:
                known_pos += 1
    return known_pos


def _decompose_step(ast, gate, diag, terminal_reason, acted_improvement):
    """Real-evidence-only failure attribution for ONE closed-loop step."""
    pos_root = sum(1 for p in gate["m2_rec"]["probe"].values() if p["g_probe"] > 0.0)
    if not gate["gated_metas"]:
        if pos_root > 0:
            return ["M2_FILTER_MISS"]   # invariant: probe positive but pool gated empty
        return ["M2_PROBE_MISS"] if diag["full_proposal_count"] > 0 else ["NO_PROPOSALS"]
    if terminal_reason in ("policy_stop", "stop_by_selector") and pos_root > 0:
        return ["M3_SELECTION_MISS"]
    if acted_improvement is not None and acted_improvement <= 0 and pos_root > 0:
        return ["M3_SELECTION_MISS"]
    if acted_improvement is not None and acted_improvement > 0:
        return ["M2_OK"]
    return ["UNKNOWN"]


def agentic_rollout_diag(env, scorer, jpol, rf, gate_mem=False, horizon=None,
                         gate_variant="r13"):
    """Replay the deterministic agentic greedy loop at rf and tag EACH step with its
    failure decomposition (M2_PROBE_MISS / M2_FILTER_MISS / M3_SELECTION_MISS /
    M2_OK / STOP_MISS) plus the §46 M2/M3/memory metric families.  Only real probe and
    execution gains count; never head-score labels.  gate_variant "r14" decomposes
    under the ADAPTIVE probe + q2 gate (R14 permanent pipeline)."""
    out = {"iid": rf["iid"], "steps": [], "decomposition": Counter(),
           "m2_metrics": {}, "final_gain": 0}
    ms0 = int(rf["schedule"].makespan)
    cache, executor = env["cache"], env["executor"]
    schedule = rf["schedule"]
    ms_cur = ms0
    _gatefn = _m2_gate_step_r14 if gate_variant == "r14" else _m2_gate_step
    for t in range(int(horizon if horizon is not None else C.TO1_R13_HORIZON)):
        prop_feats, metas, agg = cache.proposals(rf["problem"], schedule, rf["iid"])
        if not metas:
            od = {"decomp": ["NO_PROPOSALS"] + ["STOP_MISS"] if t == 0 else ["NO_PROPOSALS"],
                  "t": t}
            out["steps"].append(od)
            break
        ast = cache.ast(rf["problem"], schedule, rf["iid"])
        h = schedule_hash(schedule)
        sf = state_feature_vec(ms_cur, ms0, len(metas), agg["best_uhat"],
                               agg["best_direct"], agg["n_contrib"], agg["n_enab"])
        rng = random.Random(0)
        gate = _gatefn(ast, metas, prop_feats, jpol.m2, executor, rf["progmem"],
                       rf["iid"], rf["episode_id"], t, sf, rng)
        act = agentic_greedy_step(jpol, scorer, executor, env["model_b5"],
                                  env["single_head"], env["direct_head"],
                                  rf["problem"], schedule, rf["iid"], rf["episode_id"],
                                  rf["progmem"], ms0, step_offset=t, gate_mem=gate_mem,
                                  gate_variant=gate_variant)
        dec = _decompose_step(ast, gate, act.get("m2_diag", {}),
                              act.get("reason", "?"), act.get("improvement"))
        for d in dec:
            out["decomposition"][d] += 1
        row = {"t": t, "reason": act.get("reason"), "improvement": act.get("improvement"),
               "decomp": dec, "m2_diag": act.get("m2_diag", {}),
               "n_wide": act.get("M", 0)}
        out["steps"].append(row)
        if act["action"] != "act":
            break
        schedule = act["successor"]
        ms_cur = int(schedule.makespan)
        nh = schedule_hash(schedule)
        if nh in {schedule_hash(rf["schedule"])} and t > 0:
            break
    out["final_gain"] = ms0 - ms_cur
    return out


# ---------------------------------------------------------------------------
# §48 PERMANENT normal-M5 regression: dep-completed positive => Tier A, never Memory
# ---------------------------------------------------------------------------
def normal_m5_r13_gate(ast, metas, prop_feats, adapter, executor, pm, iid,
                       episode_id, step, sf, rng, gate_variant="r13"):
    """The permanent §48 gate trace at the DPPaulli (normal-M5) S0 state.

    Contract: for EVERY root X with direct probe <= 0 but a POSITIVE dependency-
    completed (pair) probe -- the X:M8->M5 pattern -- the filter MUST land X in
    Tier A (PROVEN_GAIN), never Tier B (Memory-only).  Memory is second-level
    evidence and can never veto real dep-completed gain.

    Returns {at_state: bool, dep_saved: [X...], all_tier_a_ok: bool, diag: ...,
              retain_ok: bool, retained: [...]}.  The entry script elevates
    all_tier_a_ok=False to the §48 stop-list.  gate_variant="r14" runs the trace on
    the ADAPTIVE gate (R14 permanent structure)."""
    gate = (_m2_gate_step_r14 if gate_variant == "r14" else _m2_gate_step)(
        ast, metas, prop_feats, adapter, executor, pm, iid,
        episode_id, step, sf, rng)
    probe = gate["m2_rec"]["probe"]
    dep_saved = sorted(o for o, p in probe.items()
                       if p["g_direct"] <= 0.0 and p["g_dep"] > 0.0)
    tiers = gate["m2_rec"]["tiers"]
    all_tier_a_ok = all(tiers.get(o) == "A" for o in dep_saved)
    retain_ok = set(dep_saved).issubset(set(gate["retained"]))
    return {"at_state": True, "dep_saved": dep_saved, "all_tier_a_ok": bool(all_tier_a_ok),
            "retain_ok": bool(retain_ok), "diag": gate["diag"],
            "retained": gate["retained"]}


# ---------------------------------------------------------------------------
# §54 cheap alpha_M3 influence diagnostic (no training)  +  §55 cloud profile
# ---------------------------------------------------------------------------
def alpha_m3_influence_diag(jpol, scorer, env, roots, gate_mem=False):
    """alpha_M3 influence WITHOUT any gradient: recompute the gated greedy action
    logits at alpha_prop=0.5 (R13 main) vs 1.0 (diagnostic escalation).  Reports raw
    margin deltas, argmax flips, and ACT-vs-STOP flips over the same states."""
    rows, n_flip, n_above0 = [], 0, 0
    base_alpha = float(jpol.m3.alpha_prop)
    try:
        jpol.m3.alpha_prop = 1.0
        for rf in roots[:12]:
            act_hi = agentic_greedy_step(jpol, scorer, env["executor"],
                                         env["model_b5"], env["single_head"],
                                         env["direct_head"], rf["problem"],
                                         rf["schedule"], rf["iid"], rf["episode_id"],
                                         rf["progmem"], int(rf["schedule"].makespan),
                                         gate_mem=gate_mem)
            jpol.m3.alpha_prop = base_alpha
            act_lo = agentic_greedy_step(jpol, scorer, env["executor"],
                                         env["model_b5"], env["single_head"],
                                         env["direct_head"], rf["problem"],
                                         rf["schedule"], rf["iid"], rf["episode_id"],
                                         rf["progmem"], int(rf["schedule"].makespan),
                                         gate_mem=gate_mem)
            lo, hi = act_lo.get("logits"), act_hi.get("logits")
            if lo is None or hi is None:
                rows.append({"iid": rf["iid"], "argmax_flip": False})
                continue
            rows.append({"iid": rf["iid"],
                         "M": int(act_lo.get("M", 0)),
                         "argmax_lo": int(lo.argmax().item()),
                         "argmax_hi": int(hi.argmax().item()),
                         "argmax_flip": int(lo.argmax().item()) != int(hi.argmax().item()),
                         "prop_above0_lo": int((lo[:-1] > lo[-1]).sum().item()),
                         "prop_above0_hi": int((hi[:-1] > hi[-1]).sum().item()),
                         "best_diff": float(hi[:-1].max() - lo[:-1].max())})
            n_flip += rows[-1]["argmax_flip"]
            n_above0 += rows[-1]["prop_above0_hi"] - rows[-1]["prop_above0_lo"]
    finally:
        jpol.m3.alpha_prop = base_alpha
    return {"rows": rows, "n_flip": n_flip, "n_states": len(rows),
            "mean_best_diff": float(np.mean([r["best_diff"] for r in rows
                                            if "best_diff" in r])) if rows else 0.0,
            "marginal_above_stop_delta": n_above0}


def _root_probe_rate(groups):
    """Root probes executed per collected group (for the §55 throughput)."""
    n_probe = 0
    n_groups = max(len(groups), 1)
    for g in groups:
        for tr in g["trajs"]:
            for rec in tr["steps"]:
                d = rec.get("m2_diag")
                if d:
                    n_probe += d["probed_root_count"]
    return n_probe / n_groups


def agentic_cloud_profile(jpol, scorer, env, root, workers=(1, 2, 4),
                          graphs=(4, 8, 16), k=8, mp_ctx=None, variant="r13",
                          action_space="full"):
    """§55 cloud/shape profile for the agentic pipeline: wall-clock per K-sibling
    group and root probes/min at workers 1/2/4 and group counts 4/8/16.  Uses the SAME
    deterministic collect as training; identical-data checks closed per worker count.
    variant="r14" profiles the ADAPTIVE-probe collect (real R14 timing); action_space
    "shortlist" (R15) profiles the SAME collect with the coverage-preserving M3 action set."""
    def _collect(*a, **kw):
        if variant == "r14":
            return collect_full_group_rollouts_r14(*a, action_space=action_space, **kw)
        return collect_full_group_rollouts_r13(*a, **kw)
    res = {}
    for w in workers:
        t0 = time.time()
        gt = _collect(
            jpol, scorer, env["executor"], env["model_b5"], env["single_head"],
            env["direct_head"], root["problem"], root["schedule"], int(root["root_ms"]),
            root["iid"], root["episode_id"], copy.deepcopy(root["progmem"]),
            k=k, seed=7, workers=w, step0_cache=env["cache"], mp_ctx=mp_ctx)
        coll_s = time.time() - t0
        # reference bit-identity: re-run w=1 with the same seed -> identical keys
        ref = _collect(
            jpol, scorer, env["executor"], env["model_b5"], env["single_head"],
            env["direct_head"], root["problem"], root["schedule"], int(root["root_ms"]),
            root["iid"], root["episode_id"], copy.deepcopy(root["progmem"]),
            k=k, seed=7, workers=1, step0_cache=env["cache"], mp_ctx=mp_ctx)
        ident = _group_identity_key(gt) == _group_identity_key(ref)
        rate = _root_probe_rate([gt])
        res[str(w)] = {"coll_s": round(coll_s, 3),
                       "identical_to_w1": bool(ident),
                       "probes_per_min": round(60 * rate * (k / coll_s), 1),
                       "throughput_trajs_per_min": round(60 * k / coll_s, 1)}
    _serial_s = res["1"]["coll_s"]
    speedups = {str(w): round(_serial_s / res[str(w)]["coll_s"], 3)
                if res[str(w)]["coll_s"] > 0 else 0.0 for w in workers}
    groups_row = {}
    per_grp = res["1"]["coll_s"]
    for ng in graphs:
        groups_row[str(ng)] = {
            "est_wall_min": round(ng * per_grp / 60, 2),
            "est_probes_per_min": round(60 * (ng * _root_probe_rate([gt])) / (ng * per_grp), 1),
        }
    return {"per_worker": res, "speedup_vs_w1": speedups, "groups": groups_row}


def graph_level_cloud_profile(jpol, scorer, env, graphs, workers=(1, 2, 4, 8),
                              k=8, horizon=None, mp_ctx=None,
                              action_space="lexicographic"):
    """Bounded T2-D worker profile on the actual graph-level collection path.

    It warms only deterministic t=0 analysis, times a complete fixed graph batch,
    checks semantic identity against serial collection and reports throughput plus
    parent/children peak-RSS counters.  It performs no policy update.
    """
    import os
    import resource

    gs = list(graphs)
    if not gs:
        raise ValueError("graph-level profile requires at least one graph")
    for g in gs:
        env["cache"].proposals(g.problem, g.schedule, g.iid)

    def _rss_mb(who):
        raw = float(resource.getrusage(who).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        return raw / (1024.0 if os.uname().sysname == "Linux" else 1024.0 ** 2)

    try:
        total_ram_mb = float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1024.0 ** 2
    except (ValueError, OSError, AttributeError):
        total_ram_mb = 0.0

    rows, ref_keys = {}, None
    for w0 in workers:
        w = max(1, min(int(w0), len(gs)))
        pool = None
        if w > 1:
            pool = make_graph_rollout_pool(
                jpol, scorer, env["executor"], env["model_b5"],
                env["single_head"], env["direct_head"], workers=w,
                mp_ctx=mp_ctx, val_cache={}, torch_threads=1)
        started = time.perf_counter()
        cpu_started = time.process_time()
        try:
            groups_out = collect_depth_groups_rollouts_r14(
                jpol, scorer, env["executor"], env["model_b5"],
                env["single_head"], env["direct_head"], gs, k=k,
                horizon=horizon, seed=7, workers=w, step0_cache=env["cache"],
                mp_ctx=mp_ctx, action_space=action_space, val_cache={},
                torch_threads=1, rollout_pool=pool)
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        elapsed = time.perf_counter() - started
        aggregate_cpu_s = (time.process_time() - cpu_started if w == 1 else
                           sum(float(g.get("worker_telemetry", {}).get("worker_cpu_s", 0.0))
                               for g in groups_out))
        keys = [_group_identity_key(g) for g in groups_out]
        if ref_keys is None:
            ref_keys = keys
        child_peak = _rss_mb(resource.RUSAGE_CHILDREN)
        estimated_pool_mb = child_peak * w
        rows[str(w0)] = {
            "effective_workers": w,
            "wall_s": elapsed,
            "graphs_per_min": 60.0 * len(gs) / max(elapsed, 1e-12),
            "trajectories_per_min": 60.0 * len(gs) * int(k) / max(elapsed, 1e-12),
            "aggregate_cpu_utilization_pct": 100.0 * aggregate_cpu_s / max(elapsed, 1e-12),
            "identical_to_w1": keys == ref_keys,
            "parent_peak_rss_mb": _rss_mb(resource.RUSAGE_SELF),
            "children_peak_rss_mb_per_process_upper_bound": child_peak,
            "estimated_pool_rss_mb_upper_bound": estimated_pool_mb,
            "ram_safe": bool(total_ram_mb <= 0 or estimated_pool_mb <= 0.8 * total_ram_mb),
        }
    valid = [(float(row["graphs_per_min"]), int(w)) for w, row in rows.items()
             if row["identical_to_w1"] and row["ram_safe"]]
    chosen = max(valid, key=lambda pair: (pair[0], -pair[1]))[1] if valid else 1
    return {"per_worker": rows, "chosen_workers": chosen,
            "n_graphs": len(gs), "K": int(k),
            "total_ram_mb": total_ram_mb,
            "note": "GPU utilization is measured during the update, not CPU rollout profile"}


# ===========================================================================
# R18 PROPOSAL-VALIDATION-GATED JOINT GRPO  (T1-PROPOSAL-VALIDATION-GATED-
# JOINT-GRPO-R18)  §0-§68
#
# CANONICAL runtime (final, §1):
#   S_t + Appearance -> M2 attribution -> M2 budgeted root probe ->
#   makespan-first / Memory-second root filtering (R14 §2 verbatim) ->
#   Reasoner -> complete legal Proposal pool ->
#   REAL complete-Proposal validation (§4: G_prop(P) = Cmax(S_t)-Cmax(S'_P) under
#   FixedDecisionReplay) -> makespan-first / Memory-second Proposal filtering
#   (§6-7, §10) -> validated Proposal pool -> M3 Proposal/STOP ->
#   FixedDecisionReplay -> S_{t+1}.
#
# M3 action space (§12-13): ALL validated Tier-A Proposals (G_prop>0, uncapped,
# Memory cannot veto §6/§8) UNION max B_PROP_MEMORY=4 Tier-B (MEMORY_RESCUED)
# Proposals UNION STOP.  NO R15/R16 shortlist32, NO extra Top-K, no cap32.
#
# ProposalProbeGain is a deterministic runtime layer observation, NOT a reward
# (§14-16, §22-24) and NOT a causal posterior (§56).  M3 selection stays on the
# frozen SFT base + zero-init ProposalEvidenceResidualAdapter (§18-20) trained by
# Joint GRPO with stagewise A2/A3 credit (§21-25).
# ===========================================================================


class M3ProposalEvidencePolicy(nn.Module):
    """R18 §18-20: frozen m3_proposal_top1_sft_v2.pt SELECTOR + small zero-init
    ProposalEvidenceResidualAdapter.

    score_M3(P) = score_SFT_base(P) + alpha_evidence * tanh(delta_evidence(P)),
    delta_evidence reads [existing M3 proposal features |
                          gain_norm | is_memory_rescued | memory_confidence]
    (EVID_DIM=3, §16/§18).  STOP = frozen R6 STOP base + small residual on
    (state, pool_stats).  alpha_evidence=1.0, zero-init => delta==0 =>
    first-forward parity with the R6 SFT on the SAME validated action set (§20).

    Interface mirrors M3RollingGRPOPolicy (`_base_raw` tuple contract,
    `action_logits`, `base_logits`, `prop_scores`, `stop_head`, snapshot) so the
    R13/R14 collect/greedy/parity/update harnesses consume it unchanged; the
    evidence tensor is the only addition (`evid=None` defaults to the R6 base)."""

    def __init__(self, r6_selector, alpha_evidence=None):
        super().__init__()
        self.r6 = r6_selector
        self.r6.eval()
        for p in self.r6.parameters():
            p.requires_grad_(False)
        self.alpha_evidence = float(alpha_evidence if alpha_evidence is not None
                                    else C.TO1_R18_ALPHA_EVIDENCE)
        self.alpha_prop = self.alpha_evidence     # harness compat (§19 keeps alpha_evidence)
        self.alpha_stop = self.alpha_evidence
        self.resid_prop = nn.Linear(277 + int(C.TO1_R18_EVID_DIM), 1)   # F_pool + evid
        self.resid_stop = nn.Linear(7 + 5, 1)     # STATE_FEAT_DIM + STOP_POOL_STAT_DIM
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
    def residual_prop(self, F_pool, evid=None):
        if F_pool is None or len(F_pool) == 0:
            return torch.zeros(0, dtype=torch.float32)
        F = F_pool.float()
        if evid is not None and len(evid) == len(F_pool):
            F = torch.cat([F, evid.float()], dim=-1)
        else:
            F = torch.cat([F, torch.zeros(
                len(F_pool), int(C.TO1_R18_EVID_DIM), dtype=torch.float32)], dim=-1)
        return float(self.alpha_evidence) * torch.tanh(
            self.resid_prop(F).squeeze(-1))                        # [M]

    def residual_stop(self, state_feat, pool_stats):
        stop_in = self._stop_in(state_feat, pool_stats)
        return float(self.alpha_evidence) * torch.tanh(
            self.resid_stop(stop_in).squeeze(-1)).reshape(1)      # [1]

    # -- score path ----------------------------------------------------------
    def forward(self, F_pool, state_feat, pool_stats=None, evid=None):
        if pool_stats is None:
            pool_stats = _pool_stats_from(F_pool)
        z, _, _ = self._base_raw(F_pool)
        base_stop = self._stop_base(state_feat, pool_stats)
        return (z + self.residual_prop(F_pool, evid),                     # [M]
                base_stop + self.residual_stop(state_feat, pool_stats))   # [1]

    def base_logits(self, F_pool, state_feat, pool_stats=None, evid=None,
                    traj_ctx=None, **_kw):
        del traj_ctx
        if pool_stats is None:
            pool_stats = _pool_stats_from(F_pool)
        z, _, _ = self._base_raw(F_pool)
        return torch.cat([z, self._stop_base(state_feat, pool_stats).reshape(1)], dim=-1)

    def action_logits(self, F_pool, state_feat, pool_stats=None, evid=None,
                      traj_ctx=None, **_kw):
        del traj_ctx
        sp, ss = self.forward(F_pool, state_feat, pool_stats, evid=evid)
        return torch.cat([sp, ss], dim=-1)         # [M+1]

    def prop_scores(self, F, evid=None):
        z, _, _ = self._base_raw(F)
        return z + self.residual_prop(F, evid)

    def stop_head(self, stop_in):
        sf = stop_in[:, :7]
        ps = stop_in[:, 7:12]
        return self._stop_base(sf.reshape(1, -1), ps.reshape(1, -1)).reshape(1, 1)

    def snapshot(self):
        with torch.no_grad():
            return {"params": {name: p.detach().clone()
                               for name, p in self.named_parameters()}}

    def load_snapshot(self, snap):
        params = snap.get("params", snap)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name in params:
                    p.copy_(params[name])


class M3TemporalEvidencePolicy(nn.Module):
    """R19 §27-29: frozen m3_proposal_top1_sft_v2.pt SELECTOR + small zero-init
    ProposalTemporalEvidenceAdapter.

    score_M3(P) = score_SFT_base(P) + alpha_temporal * tanh(delta_temporal(P)),
    delta_temporal reads [existing proposal F_pool(277) |
                         g1_norm | gh_norm | is_immediate_positive |
                         is_delayed_positive | is_memory_rescued |
                         memory_confidence]  (EVID_DIM=6, §25/§27).

    STOP = frozen R6 STOP base + small residual on (state, pool_stats).  G1/GH
    are OBSERVATIONS ONLY -- NEVER reward shaping (§33).  alpha_temporal=1.0,
    zero-init => delta==0 => first-forward bit-identical parity with the frozen
    R6 SFT on the SAME H-step-validated action set (§29).  NO new M3 SFT (§30);
    R16/R17 experimental checkpoints are banned as parents (§35).

    Interface mirrors M3RollingGRPOPolicy/M3ProposalEvidencePolicy (_base_raw
    tuple, action_logits, base_logits, prop_scores, stop_head, snapshot) so the
    joint harnesses consume it unchanged; `evid` (6-d) is the only addition."""

    def __init__(self, r6_selector, alpha_temporal=None):
        super().__init__()
        self.r6 = r6_selector
        self.r6.eval()
        for p in self.r6.parameters():
            p.requires_grad_(False)
        self.alpha_temporal = float(alpha_temporal
                                    if alpha_temporal is not None
                                    else C.TO1_R19_ALPHA_TEMPORAL)
        self.alpha_prop = self.alpha_temporal    # harness compat (§28 keeps alpha_temporal)
        self.alpha_stop = self.alpha_temporal
        self.resid_prop = nn.Linear(277 + int(C.TO1_R19_EVID_DIM), 1)  # F_pool + evid(6)
        self.resid_stop = nn.Linear(7 + 5, 1)                          # STATE_FEAT + STOP_POOL_STAT
        self._zero_residual()

    def _zero_residual(self):
        for head in (self.resid_prop, self.resid_stop):
            with torch.no_grad():
                head.weight.zero_()
                head.bias.zero_()

    def _base_raw(self, F_pool):
        if F_pool is not None and len(F_pool):
            with torch.no_grad():
                raw = self.r6.prop_scores(F_pool)   # frozen, no grad
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

    def residual_prop(self, F_pool, evid=None):
        if F_pool is None or len(F_pool) == 0:
            return torch.zeros(0, dtype=torch.float32)
        F = F_pool.float()
        if evid is not None and len(evid) == len(F_pool):
            F = torch.cat([F, evid.float()], dim=-1)
        else:
            F = torch.cat([F, torch.zeros(
                len(F_pool), int(C.TO1_R19_EVID_DIM), dtype=torch.float32)], dim=-1)
        return float(self.alpha_temporal) * torch.tanh(
            self.resid_prop(F).squeeze(-1))

    def residual_stop(self, state_feat, pool_stats):
        stop_in = self._stop_in(state_feat, pool_stats)
        return float(self.alpha_temporal) * torch.tanh(
            self.resid_stop(stop_in).squeeze(-1)).reshape(1)

    def forward(self, F_pool, state_feat, pool_stats=None, evid=None):
        if pool_stats is None:
            pool_stats = _pool_stats_from(F_pool)
        z, _, _ = self._base_raw(F_pool)
        base_stop = self._stop_base(state_feat, pool_stats)
        return (z + self.residual_prop(F_pool, evid),                     # [M]
                base_stop + self.residual_stop(state_feat, pool_stats))   # [1]

    def base_logits(self, F_pool, state_feat, pool_stats=None, evid=None,
                    traj_ctx=None, **_kw):
        del traj_ctx
        if pool_stats is None:
            pool_stats = _pool_stats_from(F_pool)
        z, _, _ = self._base_raw(F_pool)
        return torch.cat([z, self._stop_base(state_feat, pool_stats).reshape(1)], dim=-1)

    def action_logits(self, F_pool, state_feat, pool_stats=None, evid=None,
                      traj_ctx=None, **_kw):
        del traj_ctx
        sp, ss = self.forward(F_pool, state_feat, pool_stats, evid=evid)
        return torch.cat([sp, ss], dim=-1)         # [M+1]

    def prop_scores(self, F, evid=None):
        z, _, _ = self._base_raw(F)
        return z + self.residual_prop(F, evid)

    def stop_head(self, stop_in):
        sf = stop_in[:, :7]
        ps = stop_in[:, 7:12]
        return self._stop_base(sf.reshape(1, -1), ps.reshape(1, -1)).reshape(1, 1)

    def snapshot(self):
        with torch.no_grad():
            return {"params": {name: p.detach().clone()
                               for name, p in self.named_parameters()}}

    def load_snapshot(self, snap):
        params = snap.get("params", snap)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name in params:
                    p.copy_(params[name])


def proposal_validate_r18(ast, metas, prop_feats, rolex, executor, pm, iid, episode_id,
                          step, sf, state_hash, ms_cur, cache=None, mem_budget=None,
                          mem_support_min=None, mem_success_min=None,
                          result_of=None):
    """R18 §4-10 REAL complete-Proposal counterfactual validation.

    For EVERY complete legal Proposal P in `metas` (the M2-root-gated Reasoner
    pool) run FixedDecisionReplay(P) on the CURRENT state:
        G_prop(P) = Cmax(S_t) - Cmax(S'_P)     ($4 ProposalProbeGain)
    Then:
      Tier-A PROVEN_IMMEDIATE_GAIN  G_prop > 0          -> keep, uncapped (§6)
      Tier-B MEMORY_RESCUED         G_prop <= 0 AND Memory supports (§7-9)
      Tier-C PRUNE                  otherwise (incl. infeasible) (§7)
    Tier-B capped at B_PROP_MEMORY=4, ordered by (mean state-similarity weight,
    support count, historical confidence) -- NEVER future true_U (§10).

    `cache` (optional) memoizes (iid, state_hash, proposal_signature) ->
    replay result (deterministic same-runtime replay cache, §29) + returns the
    per-call lookup/hit tallies.  `result_of` (optional) injects a precomputed
    replay result for unit/parity tests (never at runtime).
    """
    import math
    N = len(metas)
    mem_budget = int(mem_budget if mem_budget is not None
                     else C.TO1_R18_PROP_MEMORY_BUDGET)
    mem_support_min = float(mem_support_min if mem_support_min is not None
                            else C.TO1_R18_MEM_SUPPORT_MIN)
    mem_success_min = float(mem_success_min if mem_success_min is not None
                            else C.TO1_R18_MEM_SUCCESS_MIN)
    cache = {} if cache is None else cache
    cache_lookups, cache_hits = 0, 0
    gmem = float(pm.retrieval_gate(iid, episode_id, step, sf))

    rows = []
    for k in range(N):
        meta = metas[k]
        sig = proposal_identity(ast, meta)[2]
        key = (iid, state_hash, sig)
        cache_lookups += 1
        if result_of is not None and result_of(key) is not None:
            res = result_of(key)
            cache_hits += 1
        elif key in cache:
            res = cache[key]
            cache_hits += 1
        else:
            edits, _kind = _edits_for(ast, meta)
            res = _execute_step(executor, ast["problem"], ast["schedule"], edits,
                                ms_cur, state_hash)
            cache[key] = res
        if res is None:
            g = float("-inf")
            valid = False
        else:
            g = float(res["improvement"])
            valid = True
        rows.append({"k": k, "sig": sig, "kind": meta["kind"], "g": g, "valid": valid,
                     "tier": None, "is_mem": 0, "conf": 0.0, "gain_norm": 0.0,
                     "mem_support": 0.0, "mem_success": 0.0, "mem_mean_gain": 0.0,
                     "n_exact": 0, "n_fine": 0, "mean_w": 0.0, "mem_exact": False})

    # -- Memory consulted ONLY for valid G_prop <= 0 Proposals (§7) ---------
    for r in rows:
        if r["g"] > 0.0:
            r["tier"] = "A"
            continue
        if not r["valid"]:
            r["tier"] = "C"
            continue
        r["tier"] = "C"                      # provisional; B needs memory support
        me = pm.proposal_mem_evidence(iid, episode_id, step, sf, r["sig"],
                                      rolex["type"][r["k"]], rolex["role"][r["k"]],
                                      rolex["src"][r["k"]], rolex["tgt"][r["k"]])
        r["mem_support"], r["mem_success"], r["mem_mean_gain"] = \
            me["support"], me["success"], me["mean_gain"]
        r["n_exact"], r["n_fine"], r["mean_w"], r["mem_exact"] = \
            me["n_exact"], me["n_fine"], me["mean_w"], me["exact"]
        if (r["mem_support"] >= mem_support_min and
                r["mem_success"] >= mem_success_min and
                gmem > 0.0):
            r["tier"] = "B"
            r["conf"] = r["mem_support"] / (r["mem_support"] + float(C.TO1_R18_MEM_CONF_SCALE))
            r["is_mem"] = 1

    # -- gain normalization (§17): log1p positive -> /max positive log-gain ----
    gpos = [math.log1p(max(r["g"], 0.0)) for r in rows]
    max_gpos = max(gpos) if gpos else 0.0
    div = max_gpos + 1e-9
    for r in rows:
        if r["g"] > 0.0:
            r["gain_norm"] = math.log1p(r["g"]) / div
        # memory-rescued non-positive stays gain_norm=0 (§17)

    # -- Tier-B cap by (state-similarity, support, confidence); NEVER true_U ---
    b_cands = sorted([r for r in rows if r["tier"] == "B"],
                     key=lambda r: (-r["mean_w"], -r["mem_support"], -r["conf"]))
    b_keep = {r["k"] for r in b_cands[:mem_budget]}
    for r in rows:
        if r["tier"] == "B" and r["k"] not in b_keep:
            r["tier"] = "C"

    validated = sorted([r["k"] for r in rows if r["tier"] in ("A", "B")])
    n_pos = sum(1 for r in rows if r["g"] > 0.0)
    n_zero = sum(1 for r in rows if r["valid"] and r["g"] == 0.0)
    n_neg = sum(1 for r in rows if r["valid"] and r["g"] < 0.0)
    n_inf = sum(1 for r in rows if not r["valid"])
    n_a = sum(1 for r in rows if r["tier"] == "A")
    n_b = sum(1 for r in rows if r["tier"] == "B")
    density = (n_pos / N) if N else 0.0
    diag = {
        "full_pool_count": N, "positive_count": n_pos, "zero_count": n_zero,
        "negative_count": n_neg, "infeasible_count": n_inf,
        "tier_A": n_a, "tier_B": n_b, "tier_C": N - n_a - n_b,
        "memory_rescued_count": n_b, "validated_count": len(validated),
        "positive_density": density,
        "cache_lookups": cache_lookups, "cache_hits": cache_hits,
        "feasible": N - n_inf,
    }
    return {"validated_idx": validated, "rows": rows, "diag": diag,
            "gmem": gmem, "mem_budget": mem_budget}


def build_validated_action_set_r18(ast, gate, rolex, scorer, executor, pm, iid,
                                   episode_id, step, sf, state_hash, ms_cur, mem_sel,
                                   cache=None, mem_budget=None):
    """Wires the M2-gated Reasoner pool into the R18 M3 action set:

        validated_idx (index into gate['gated_metas']) + F_pool [M,277] +
        evid [M,EVID_DIM] (gain_norm/is_memory_rescued/confidence, §16) +
        pool_stats + validated_pool_signature (§51) + per-call pval diag.

    STOP = index M.  NEVER the R15/R16 shortlist and never an extra Top-K
    (§12-13).  Deterministic: the whole path is pure (state, memory, policy).
    """
    pv = proposal_validate_r18(ast, gate["gated_metas"], gate["gated_prop_feats"],
                               rolex, executor, pm, iid, episode_id, step, sf,
                               state_hash, ms_cur, cache=cache, mem_budget=mem_budget)
    pool = pv["validated_idx"]
    if not pool:
        return {"pool": [], "F_pool": None, "evid": None, "pool_stats": None,
                "validated_pool_signature": None, "signatures": [], "pval": pv}
    F_all = _rerank_feats_all(scorer, rolex, mem_sel)
    if F_all is None or len(F_all) == 0:
        return {"pool": [], "F_pool": None, "evid": None, "pool_stats": None,
                "validated_pool_signature": None, "signatures": [], "pval": pv}
    F_pool = F_all[pool]
    row_by_k = {r["k"]: r for r in pv["rows"]}
    # pool order stable: map by k (§51 old/new action-set identity on signatures)
    evid_rows = [[float(row_by_k[k]["gain_norm"]), float(row_by_k[k]["is_mem"]),
                  float(row_by_k[k]["conf"])] for k in pool]
    evid = torch.tensor(evid_rows, dtype=torch.float32)
    pool_stats = _pool_stats_from(F_pool)
    sigs = [row_by_k[k]["sig"] for k in pool]
    validated_pool_signature = _shortlist_sig_hash(sigs)
    return {"pool": pool, "F_pool": F_pool, "evid": evid, "pool_stats": pool_stats,
            "validated_pool_signature": validated_pool_signature,
            "signatures": sigs, "pval": pv}


# ---------------------------------------------------------------------------
# R19 -- immediate evidence -> bounded H-step rescue -> Memory fallback  (§0-24)
# ---------------------------------------------------------------------------
def _r19_cont_eval(cache, executor, cont_m2, cont_m3, scorer, problem, schedule,
                   iid, episode_id, base_ms, bpm, rng, hcache=None):
    """ONE deterministic FROZEN continuation step (R19 §13-15, §40).

    At `schedule` (a BRANCH state inside multi-step validation): run the canonical
    M2 gate (`_m2_gate_step_r14`) with a FROZEN M2 adapter, build the complete
    legal pool, score with the FROZEN canonical M3 SFT selector (δ=0, evid=None),
    argmax Proposal/STOP, execute.  Returns (schedule', ms') on a REAL executed
    positive improvement; None on STOP / empty pool / infeasible / non-positive --
    matching the canonical closed-loop terminal semantics (§15).  Reads ONLY
    frozen weights + branch-local memory; NEVER the trainable jpol and NEVER
    persistent Memory (§13/§14/§40/§41)."""
    prop_feats, metas, agg = cache.proposals(problem, schedule, iid)
    n_prop = len(metas)
    if n_prop == 0:
        return None
    ast = cache.ast(problem, schedule, iid)
    h = schedule_hash(schedule)
    ms_cur = int(schedule.makespan)
    sf = state_feature_vec(ms_cur, base_ms, n_prop, agg["best_uhat"], agg["best_direct"],
                           agg["n_contrib"], agg["n_enab"])
    sf_t = torch.tensor(sf, dtype=torch.float32)
    gate = _m2_gate_step_r14(ast, metas, prop_feats, cont_m2, executor, bpm,
                             iid, episode_id, 0, sf, rng)
    if not gate["gated_metas"]:
        return None
    rolex = _rollex_of(ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem0 = torch.zeros(len(queries), MEM_FEAT_DIM, dtype=torch.float32)
    F_all = _rerank_feats_all(scorer, rolex, mem0)          # [N, 277] frozen pipeline
    if F_all is None or len(F_all) == 0:
        return None
    pool_stats = _pool_stats_from(F_all)
    with torch.no_grad():
        logits = selector_action_logits(cont_m3, F_all, sf_t, pool_stats, evid=None)
    sel = int(logits.argmax().item())
    if sel == len(F_all):                                     # frozen policy STOP
        return None
    edits, _kind = _edits_for(ast, gate["gated_metas"][sel])
    res = _execute_step(executor, problem, schedule, edits, ms_cur, h)
    if res is None or res["improvement"] <= 0:
        return None
    return res["schedule"], int(res["schedule"].makespan)


def _r19_ghog1(acache, executor, cont_m2, cont_m3, scorer, ast, problem, schedule,
               iid, episode_id, step, sf, state_hash, ms_cur, row, sig, rng,
               res=None, hcache=None, h_val=None, result_of=None, policy_id=None):
    """Bounded H-step delayed-gain evaluation for ONE proposal P (§10-17).

    First action FIXED to P (§12) on a branch copy of the CURRENT schedule, then
    up to H continuation steps under the FROZEN continuation policy (§13).
    Returns (g1_forced, gh, term_hash).  If P's forced first step is infeasible,
    gh=-inf.  `res` is the ALREADY-COMPUTED G1 FixedDecisionReplay result (the
    forced-first-step is bit-identical to the G1 replay, §29/§52-53); None means
    infeasible.  `h_val` defaults to config TO1_R19_H_VAL (= TO1_R13_HORIZON = 5,
    §10; user 2026-08-30 lifted the §10/§11 H=2 cap in favor of the horizon that
    all previous JOINT rounds used); the §65 diagnostic samples a LONGER frozen
    continuation (h_val > H_VAL) to measure beyond-validation miss -- never used
    for runtime classification.  `policy_id` defaults to R19's; R20 passes its
    own `TO1_R20_VALIDATION_POLICY_ID` + `h_val=TO1_R20_H_VAL` so the frozen
    continuation horizon and cache key are the R20 ones (§6/§7, H_VAL=2 fixed).
    `hcache` memoizes
    (iid, state_hash, sig, policy_id, H)-keyed BASE-DERIVED (G1, GH, term_hash)
    only (§37/§38) -- Memory assessment stays time-dependent and is always
    re-computed by the caller (§19-20).  `acache` is the AnalyzeCache used to
    re-derive branch-state pools (§40 branch isolation)."""
    H = int(h_val if h_val is not None else C.TO1_R19_H_VAL)
    policy_id = str(policy_id if policy_id is not None else C.TO1_R19_VALIDATION_POLICY_ID)
    key = (iid, state_hash, sig, policy_id, H)
    if result_of is not None and result_of(key) is not None:
        return result_of(key)
    if hcache is not None and key in hcache:
        return hcache[key]
    bpm = ProgressiveMemory(state_feats=[sf])
    rng = rng if rng is not None else random.Random(_traj_seed(0, iid, episode_id, 0, 0))
    res0 = res
    if res0 is None:
        out = (row["g1"], float("-inf"), None)
    else:
        g_force = float(res0["improvement"])
        b_sch, b_ms = res0["schedule"], int(res0["schedule"].makespan)
        rng_step = random.Random(int(rng.random() * 2**31)) if hasattr(rng, "random") else rng
        for _ in range(H):
            nxt = _r19_cont_eval(acache, executor, cont_m2, cont_m3, scorer, problem,
                                 b_sch, iid, episode_id, ms_cur, bpm, rng_step)
            if nxt is None:
                break
            b_sch, b_ms = nxt
        gh = float(ms_cur - b_ms)
        out = (g_force, gh, schedule_hash(b_sch))
    if hcache is not None:
        hcache[key] = out
    return out


def proposal_validate_r19(ast, metas, prop_feats, rolex, executor, pm, iid, episode_id,
                          step, sf, state_hash, ms_cur, cont_m2, cont_m3, scorer,
                          acache=None, rc=None, hcache=None, mem_budget=None,
                          mem_support_min=None, mem_success_min=None, result_of=None):
    """R19 §4-24 REAL H-step counterfactual Proposal validation.

    For EVERY complete legal Proposal P:
      G1 = Cmax(S_t) - Cmax(S'_P)                         FixedDecisionReplay
        G1 > 0                -> class PA IMMEDIATE_POSITIVE   (§6-7, uncapped §21)
        G1 <= 0               -> bounded continuation (§8-10, §12-13), up to
                                 H_VAL (=5 = the prior JOINT horizon) frozen steps
            GH > 0            -> class PB DELAYED_POSITIVE     (§16-17, uncapped §21)
            GH <= 0           -> Memory (immediate > delayed > memory order, §19)
                strong support -> class PC MEMORY_RESCUED (cap B_PROP_MEMORY=4, §20)
                else          -> class PD UNSUPPORTED/prune    (§18)
    Memory consulted ONLY for (G1<=0 AND GH<=0).  G1 is memoized in `rc`
    (R18-style (iid, state_hash, sig) replay memo), G1+GH+term_hash in `hcache`
    ((...​, validation_policy_id, H_VAL), §37/§38).  Memory tier decisions
    are time-dependent and always fresh (§19).  `acache` = AnalyzeCache for
    branch-state pool re-derivation (§40).  `result_of` injects precomputed
    replay results for unit/parity tests only (never at runtime)."""
    import math
    N = len(metas)
    mem_budget = int(mem_budget if mem_budget is not None else C.TO1_R19_PROP_MEMORY_BUDGET)
    mem_support_min = float(mem_support_min if mem_support_min is not None
                            else C.TO1_R19_MEM_SUPPORT_MIN)
    mem_success_min = float(mem_success_min if mem_success_min is not None
                            else C.TO1_R19_MEM_SUCCESS_MIN)
    rc = {} if rc is None else rc
    hcache = {} if hcache is None else hcache
    gmem = float(pm.retrieval_gate(iid, episode_id, step, sf))

    problem, schedule = ast["problem"], ast["schedule"]
    rng = random.Random(int(_traj_seed(0, iid, episode_id, state_hash, 0)))

    rows = []
    for k in range(N):
        meta = metas[k]
        sig = proposal_identity(ast, meta)[2]
        # -- G1 via the same single-step replay memo as R18 --------------------
        key1 = (iid, state_hash, sig)
        if result_of is not None and result_of(key1) is not None:
            res = result_of(key1)
        elif key1 in rc:
            res = rc[key1]
        else:
            edits, _kind = _edits_for(ast, meta)
            res = _execute_step(executor, problem, schedule, edits, ms_cur, state_hash)
            rc[key1] = res
        valid = res is not None
        g1 = float(res["improvement"]) if valid else float("-inf")
        row = {"k": k, "sig": sig, "kind": meta["kind"], "meta": meta,
               "g1": g1, "valid": valid, "class": None, "gh": g1,
               "is_immediate": 0, "is_delayed": 0, "is_mem": 0, "conf": 0.0,
               "gain_norm1": 0.0, "gain_normH": 0.0,
               "mem_support": 0.0, "mem_success": 0.0,
               "n_exact": 0, "n_fine": 0, "mean_w": 0.0, "mem_exact": False}
        if g1 > 0.0:
            row["class"], row["is_immediate"] = "PA", 1
        elif not valid:
            row["class"] = "PD"
        else:
            # -- bounded multi-step delayed-gain (branch copy, frozen policy) --
            # `res` is the already-computed G1 replay -- the forced first step
            # MUST be bit-identical to it (§29/§52-53), never a re-execution.
            g_force, gh, _th = _r19_ghog1(
                acache, executor, cont_m2, cont_m3, scorer, ast, problem, schedule,
                iid, episode_id, step, sf, state_hash, ms_cur, row, sig, rng,
                res=res, hcache=hcache, result_of=result_of)
            row["gh"] = gh
            if gh > 0.0:
                row["class"], row["is_delayed"] = "PB", 1
            else:
                me = pm.proposal_mem_evidence(iid, episode_id, step, sf, sig,
                                              rolex["type"][k], rolex["role"][k],
                                              rolex["src"][k], rolex["tgt"][k])
                row["mem_support"], row["mem_success"], _mmg = \
                    me["support"], me["success"], me["mean_gain"]
                row["n_exact"], row["n_fine"], row["mean_w"], row["mem_exact"] = \
                    me["n_exact"], me["n_fine"], me["mean_w"], me["exact"]
                if (row["mem_support"] >= mem_support_min and
                        row["mem_success"] >= mem_success_min and gmem > 0.0):
                    row["class"], row["is_mem"] = "PC", 1
                    row["conf"] = row["mem_support"] / (row["mem_support"]
                                                        + float(C.TO1_R19_MEM_CONF_SCALE))
                else:
                    row["class"] = "PD"
        rows.append(row)

    # -- normalization (§26): state-internal signed/log-bounded in [-1, 1] -----
    gp1 = [math.log1p(max(r["g1"], 0.0)) for r in rows]
    max1 = max(gp1) if gp1 else 0.0
    den1 = max1 + 1e-9
    gpH = [math.log1p(max(r["gh"], 0.0)) for r in rows]
    maxH = max(gpH) if gpH else 0.0
    denH = maxH + 1e-9
    for r in rows:
        if r["g1"] > 0.0:
            r["gain_norm1"] = math.log1p(r["g1"]) / den1
        if r["gh"] > 0.0:
            r["gain_normH"] = math.log1p(r["gh"]) / denH

    # -- PC (MEMORY_RESCUED) cap by (state-similarity, support, confidence) ----
    m_cands = sorted([r for r in rows if r["class"] == "PC"],
                     key=lambda r: (-r["mean_w"], -r["mem_support"], -r["conf"]))
    m_keep = {r["k"] for r in m_cands[:mem_budget]}
    for r in rows:
        if r["class"] == "PC" and r["k"] not in m_keep:
            r["class"] = "PD"

    validated = sorted([r["k"] for r in rows if r["class"] in ("PA", "PB", "PC")])
    n_pos1 = sum(1 for r in rows if r["g1"] > 0.0)
    n_delayed = sum(1 for r in rows if r["is_delayed"])
    n_mem = sum(1 for r in rows if r["class"] == "PC")
    n_pruned = sum(1 for r in rows if r["class"] == "PD")
    n_inf = sum(1 for r in rows if not r["valid"])
    n_nonpos = N - n_pos1 - n_inf               # N(G1<=0 and feasible)  §46 denom
    # §46 DelayedRescueRate = N(G1<=0 ∧ GH>0) / N(G1<=0)  (feasible non-positives)
    delayed_rescue_rate = (n_delayed / n_nonpos) if n_nonpos else 0.0
    diag = {
        "full_pool_count": N, "g1_positive_count": n_pos1,
        "delayed_positive_count": n_delayed,
        "memory_rescued_count": n_mem, "pruned_count": n_pruned,
        "infeasible_count": n_inf, "validated_count": len(validated),
        "delayed_rescue_rate": delayed_rescue_rate,
        "gmem": gmem,
    }
    return {"validated_idx": validated, "rows": rows, "diag": diag,
            "mem_budget": mem_budget, "h_val": int(C.TO1_R19_H_VAL)}


def build_multistep_action_set_r19(ast, gate, rolex, scorer, executor, pm, iid,
                                   episode_id, step, sf, state_hash, ms_cur, mem_sel,
                                   cont_m2, cont_m3, acache=None, rc=None, hcache=None,
                                   mem_budget=None, result_of=None):
    """R19 §22 M3 action set on the H-step-validated pool:

        ALL IMMEDIATE_POSITIVE ∪ ALL DELAYED_POSITIVE ∪ MAX-4 MEMORY_RESCUED ∪ STOP
      (NO cap32, NO extra Top-K -- §21/§22).

    `evid` [M,6] = (g1_norm, gh_norm, is_immediate, is_delayed, is_mem, conf,
    §25/§26) is the OBSERVATION M3's temporal adapter may see -- nothing more.
    `pool_sig_hash` (§60) covers (proposal signatures + class + g1 + gh + flags)
    so an E=3/old-vs-new pool change is detectable.  `acache` = AnalyzeCache for
    the bounded continuation branch (same as the caller's env cache)."""
    pv = proposal_validate_r19(
        ast, gate["gated_metas"], gate["gated_prop_feats"], rolex, executor, pm,
        iid, episode_id, step, sf, state_hash, ms_cur, cont_m2, cont_m3, scorer,
        acache=acache, rc=rc, hcache=hcache, mem_budget=mem_budget,
        result_of=result_of)
    pool = pv["validated_idx"]
    if not pool:
        return {"pool": [], "F_pool": None, "evid": None, "pool_stats": None,
                "evid_rows": [], "pool_sig_hash": None, "signatures": [],
                "pval": pv}
    F_all = _rerank_feats_all(scorer, rolex, mem_sel)
    if F_all is None or len(F_all) == 0:
        return {"pool": [], "F_pool": None, "evid": None, "pool_stats": None,
                "evid_rows": [], "pool_sig_hash": None, "signatures": [],
                "pval": pv}
    F_pool = F_all[pool]
    row_by_k = {r["k"]: r for r in pv["rows"]}
    # §60 old/new candidate identity: stable pool order (by k) + class + gains
    evid_rows = [
        [float(row_by_k[k]["gain_norm1"]), float(row_by_k[k]["gain_normH"]),
         float(row_by_k[k]["is_immediate"]), float(row_by_k[k]["is_delayed"]),
         float(row_by_k[k]["is_mem"]), float(row_by_k[k]["conf"])] for k in pool]
    evid = torch.tensor(evid_rows, dtype=torch.float32)
    pool_stats = _pool_stats_from(F_pool)
    sigs = [row_by_k[k]["sig"] for k in pool]
    pool_sig_hash = _shortlist_sig_hash(
        tuple(f"{s}::{row_by_k[k]['class']}::{int(row_by_k[k]['g1'])}::"
              f"{int(row_by_k[k]['gh'])}::{'M' if row_by_k[k]['is_mem'] else 'N'}"
              for k in pool for s in [row_by_k[k]["sig"]]))
    return {"pool": pool, "F_pool": F_pool, "evid": evid, "evid_rows": evid_rows,
            "pool_stats": pool_stats, "pool_sig_hash": pool_sig_hash,
            "signatures": sigs, "pval": pv}


def proposal_validate_r20(ast, metas, prop_feats, rolex, executor, pm, iid, episode_id,
                          step, sf, state_hash, ms_cur, cont_m2, cont_m3, scorer,
                          acache=None, rc=None, hcache=None, mem_budget=None,
                          mem_support_min=None, mem_success_min=None, result_of=None,
                          feasible_fallback=False, feasible_fallback_cap=4):
    """R20 §3-11 STATE-LEVEL LEXICOGRAPHIC Proposal validation.

    G1 = Cmax(S_t)-Cmax(S'_P) via FixedDecisionReplay for EVERY complete legal
    Proposal (memoized in `rc`, R18-style).  Then the state-level hierarchy (§12):

        PA non-empty  -> M3 = PA ∪ STOP          (GH NOT computed §4/§15)
        else PB non-empty -> M3 = PB ∪ STOP      (GH computed only here §5/§8)
        else PC non-empty -> M3 = max4 PC ∪ STOP (Memory only here §10/§11)
        else -> M3 = {STOP}                      (TYPE-IV §11)

    No score mixing (§13): G1/GH/Memory never summed into a reward or score; the
    hierarchy is expressed ONLY through action-set eligibility.  `gh_calls`
    (§15/§41) counts GH continuation evals (0 for a PA-nonempty state);
    `pb_redundant` (§26) = feasible non-positive candidates in a PA-nonempty
    state (would-be delayed candidates R19 would GH-validate, R20 excludes).
    `result_of` injects precomputed replay results for unit/parity tests only.
    """
    import math
    N = len(metas)
    mem_budget = int(mem_budget if mem_budget is not None else C.TO1_R20_PROP_MEMORY_BUDGET)
    mem_support_min = float(mem_support_min if mem_support_min is not None
                            else C.TO1_R20_MEM_SUPPORT_MIN)
    mem_success_min = float(mem_success_min if mem_success_min is not None
                            else C.TO1_R20_MEM_SUCCESS_MIN)
    rc = {} if rc is None else rc
    hcache = {} if hcache is None else hcache
    gmem = float(pm.retrieval_gate(iid, episode_id, step, sf))

    problem, schedule = ast["problem"], ast["schedule"]
    rng = random.Random(int(_traj_seed(0, iid, episode_id, state_hash, 0)))
    g1_cache_lookups = g1_cache_hits = 0
    gh_cache_lookups = gh_cache_hits = 0

    # ---- phase 1: G1 for EVERY proposal (memoized replay, R18 §4) -----------
    rows = []
    nonfinite_g1_count = 0
    for k in range(N):
        meta = metas[k]
        sig = proposal_identity(ast, meta)[2]
        key1 = (iid, state_hash, sig)
        g1_cache_lookups += 1
        if result_of is not None and result_of(key1) is not None:
            res = result_of(key1)
            g1_cache_hits += 1
        elif key1 in rc:
            res = rc[key1]
            g1_cache_hits += 1
        else:
            edits, _kind = _edits_for(ast, meta)
            res = _execute_step(executor, problem, schedule, edits, ms_cur, state_hash)
            rc[key1] = res
        raw_g1 = float(res["improvement"]) if res is not None else float("-inf")
        # A malformed/corrupt replay must never enter ranking, normalization or
        # an action signature as +/-inf/NaN.  Treat it exactly like an invalid
        # replay and expose the count in diagnostics.
        if res is not None and not math.isfinite(raw_g1):
            nonfinite_g1_count += 1
        valid = res is not None and math.isfinite(raw_g1)
        g1 = raw_g1 if valid else float("-inf")
        rows.append({"k": k, "sig": sig, "kind": meta["kind"], "meta": meta,
                     "g1": g1, "valid": valid, "gh": 0.0, "class": None,
                     "is_immediate": 0, "is_delayed": 0, "is_mem": 0, "conf": 0.0,
                     "gain_norm1": 0.0, "gain_normH": 0.0,
                     "mem_support": 0.0, "mem_success": 0.0,
                     "n_exact": 0, "n_fine": 0, "mean_w": 0.0, "mem_exact": False})

    n_pos1 = sum(1 for r in rows if r["g1"] > 0.0)
    n_inf = sum(1 for r in rows if not r["valid"])
    n_nonpos_feasible = sum(1 for r in rows if r["valid"] and r["g1"] <= 0.0)
    gh_calls = 0
    nonfinite_gh_count = 0

    if n_pos1 > 0:
        # ---- TYPE-I: PA layer active (§4).  NO GH, NO Memory. --------------
        active_layer, state_type = "PA", "TYPE-I"
        for r in rows:
            if r["g1"] > 0.0:
                r["class"], r["is_immediate"] = "PA", 1
            else:
                r["class"] = "PD"
        pb_redundant = n_nonpos_feasible          # §26 upper bound (GH skipped)
    else:
        # ---- TYPE-II+: all G1<=0.  Lazy GH continuation (§5). ---------------
        for r in rows:
            if not r["valid"]:
                r["class"] = "PD"
                continue
            gh_key = (iid, state_hash, r["sig"],
                      str(C.TO1_R20_VALIDATION_POLICY_ID), int(C.TO1_R20_H_VAL))
            gh_cache_lookups += 1
            if gh_key in hcache:
                gh_cache_hits += 1
            g_force, gh, _th = _r19_ghog1(
                acache, executor, cont_m2, cont_m3, scorer, ast, problem, schedule,
                iid, episode_id, step, sf, state_hash, ms_cur, r, r["sig"], rng,
                res=rc.get((iid, state_hash, r["sig"])), hcache=hcache,
                h_val=int(C.TO1_R20_H_VAL), policy_id=str(C.TO1_R20_VALIDATION_POLICY_ID),
                result_of=result_of)
            gh_calls += 1
            if not math.isfinite(float(gh)):
                nonfinite_gh_count += 1
                gh = 0.0
            r["gh"] = float(gh)
            if gh > 0.0:
                r["class"], r["is_delayed"] = "PB", 1
        n_delayed = sum(1 for r in rows if r["is_delayed"])
        if n_delayed > 0:
            # ---- TYPE-II: PB layer active (§8/§9). -------------------------
            active_layer, state_type = "PB", "TYPE-II"
            for r in rows:
                if not r["is_delayed"]:
                    r["class"] = "PD"
            pb_redundant = 0
        else:
            # ---- Memory fallback (§10/§11) only here -----------------------
            for r in rows:
                if r["valid"] and r["class"] is None:
                    me = pm.proposal_mem_evidence(
                        iid, episode_id, step, sf, r["sig"],
                        rolex["type"][r["k"]], rolex["role"][r["k"]],
                        rolex["src"][r["k"]], rolex["tgt"][r["k"]])
                    r["mem_support"], r["mem_success"], _mmg = \
                        me["support"], me["success"], me["mean_gain"]
                    r["n_exact"], r["n_fine"], r["mean_w"], r["mem_exact"] = \
                        me["n_exact"], me["n_fine"], me["mean_w"], me["exact"]
                    if (r["mem_support"] >= mem_support_min and
                            r["mem_success"] >= mem_success_min and gmem > 0.0):
                        r["class"], r["is_mem"] = "PC", 1
                        r["conf"] = r["mem_support"] / (r["mem_support"]
                                                        + float(C.TO1_R20_MEM_CONF_SCALE))
                    else:
                        r["class"] = "PD"
            m_cands = sorted([r for r in rows if r["class"] == "PC"],
                             key=lambda r: (-r["mean_w"], -r["mem_support"], -r["conf"]))
            m_keep = {r["k"] for r in m_cands[:mem_budget]}
            for r in rows:
                if r["class"] == "PC" and r["k"] not in m_keep:
                    r["class"] = "PD"
            if m_keep:
                active_layer, state_type = "PC", "TYPE-III"
            elif feasible_fallback:
                # T2-F exploration fallback: infeasible proposals remain excluded,
                # but a feasible non-improving move is not the same thing as an
                # invalid schedule.  Keep the least damaging candidates so a
                # 12-step trajectory can cross a local makespan barrier instead
                # of collapsing into the learned STOP action.
                feasible = sorted(
                    (r for r in rows if r["valid"]),
                    key=lambda r: (-float(r["g1"]), r["sig"]))
                keep = {r["k"] for r in feasible[:max(1, int(feasible_fallback_cap))]}
                for r in rows:
                    if r["k"] in keep:
                        r["class"] = "PF"
                    elif r["class"] is None:
                        r["class"] = "PD"
                active_layer, state_type = "PF", "TYPE-IV-FEASIBLE"
            else:
                active_layer, state_type = "STOP", "TYPE-IV"
            pb_redundant = 0

    # ---- normalization (§16): state-internal signed/log-bounded in [-1, 1] --
    gp1 = [math.log1p(max(r["g1"], 0.0)) for r in rows]
    den1 = (max(gp1) if gp1 else 0.0) + 1e-9
    gpH = [math.log1p(max(r["gh"], 0.0)) for r in rows]
    denH = (max(gpH) if gpH else 0.0) + 1e-9
    for r in rows:
        if r["g1"] > 0.0:
            r["gain_norm1"] = math.log1p(r["g1"]) / den1
        if r["gh"] > 0.0:
            r["gain_normH"] = math.log1p(r["gh"]) / denH

    validated = sorted([r["k"] for r in rows
                        if r["class"] in ("PA", "PB", "PC", "PF")])
    n_mem = sum(1 for r in rows if r["class"] == "PC")
    n_pruned = sum(1 for r in rows if r["class"] == "PD")
    n_delayed = sum(1 for r in rows if r["is_delayed"])
    diag = {
        "full_pool_count": N, "g1_positive_count": n_pos1,
        "delayed_positive_count": n_delayed,
        "memory_rescued_count": n_mem, "pruned_count": n_pruned,
        "infeasible_count": n_inf, "validated_count": len(validated),
        "feasible_fallback_count": sum(1 for r in rows if r["class"] == "PF"),
        "active_layer": active_layer, "state_type": state_type,
        "gh_calls": gh_calls, "pb_redundant": pb_redundant,
        "cache_lookups": g1_cache_lookups + gh_cache_lookups,
        "cache_hits": g1_cache_hits + gh_cache_hits,
        "g1_cache_lookups": g1_cache_lookups, "g1_cache_hits": g1_cache_hits,
        "gh_cache_lookups": gh_cache_lookups, "gh_cache_hits": gh_cache_hits,
        "gmem": gmem,
        "nonfinite_g1_count": nonfinite_g1_count,
        "nonfinite_gh_count": nonfinite_gh_count,
    }
    return {"validated_idx": validated, "rows": rows, "diag": diag,
            "mem_budget": mem_budget, "h_val": int(C.TO1_R20_H_VAL)}


def build_lexicographic_action_set_r20(ast, gate, rolex, scorer, executor, pm, iid,
                                       episode_id, step, sf, state_hash, ms_cur, mem_sel,
                                       cont_m2, cont_m3, acache=None, rc=None, hcache=None,
                                       mem_budget=None, result_of=None,
                                       feasible_fallback=False, feasible_fallback_cap=4):
    """R20 §4-12 M3 action set on the state-level lexicographic fallback pool:

        PA non-empty -> PA ∪ STOP
        else PB non-empty -> PB ∪ STOP
        else PC non-empty -> max4 PC ∪ STOP
        else -> {STOP}

    `active_layer`/`state_type`/`gh_calls`/`pb_redundant` flow in `pval.diag`
    (§24-26/§41).  `evid` [M,6] is the OBSERVATION M3's temporal adapter may see
    (§16) -- nothing more.  `pool_sig_hash` covers active_layer + signatures +
    class + gains so an E=3/old-vs-new pool change is detectable (§44)."""
    pv = proposal_validate_r20(
        ast, gate["gated_metas"], gate["gated_prop_feats"], rolex, executor, pm,
        iid, episode_id, step, sf, state_hash, ms_cur, cont_m2, cont_m3, scorer,
        acache=acache, rc=rc, hcache=hcache, mem_budget=mem_budget,
        result_of=result_of, feasible_fallback=feasible_fallback,
        feasible_fallback_cap=feasible_fallback_cap)
    pool = pv["validated_idx"]
    if not pool:
        return {"pool": [], "F_pool": None, "evid": None, "pool_stats": None,
                "evid_rows": [], "pool_sig_hash": None, "signatures": [],
                "active_layer": pv["diag"]["active_layer"], "pval": pv}
    F_all = _rerank_feats_all(scorer, rolex, mem_sel)
    if F_all is None or len(F_all) == 0:
        return {"pool": [], "F_pool": None, "evid": None, "pool_stats": None,
                "evid_rows": [], "pool_sig_hash": None, "signatures": [],
                "active_layer": pv["diag"]["active_layer"], "pval": pv}
    F_pool = F_all[pool]
    row_by_k = {r["k"]: r for r in pv["rows"]}
    evid_rows = [
        [float(row_by_k[k]["gain_norm1"]), float(row_by_k[k]["gain_normH"]),
         float(row_by_k[k]["is_immediate"]), float(row_by_k[k]["is_delayed"]),
         float(row_by_k[k]["is_mem"]), float(row_by_k[k]["conf"])] for k in pool]
    evid = torch.tensor(evid_rows, dtype=torch.float32)
    pool_stats = _pool_stats_from(F_pool)
    sigs = [row_by_k[k]["sig"] for k in pool]
    layer = pv["diag"]["active_layer"]
    def _gain_token(value):
        value = float(value)
        return str(int(value)) if math.isfinite(value) else "NONFINITE"

    pool_sig_hash = _shortlist_sig_hash(
        tuple(f"{layer}::{s}::{row_by_k[k]['class']}::{_gain_token(row_by_k[k]['g1'])}::"
              f"{_gain_token(row_by_k[k]['gh'])}::{'M' if row_by_k[k]['is_mem'] else 'N'}"
              for k in pool for s in [row_by_k[k]["sig"]]))
    return {"pool": pool, "F_pool": F_pool, "evid": evid, "evid_rows": evid_rows,
            "pool_stats": pool_stats, "pool_sig_hash": pool_sig_hash,
            "signatures": sigs, "active_layer": layer, "pval": pv}


def _decompose_proposal_step_r18(gate, pv, terminal_reason, acted_improvement):
    """R18-real-evidence-only failure attribution for ONE closed-loop step under the
    validated action set.  M3_SELECTION_MISS is charged ONLY when a PROVEN-immediate
    (Tier-A, G_prop>0) Proposal was in the validated set but M3 stopped / acted
    non-positive.  A correct STOP on an empty or Tier-B-only validated set is
    NOT a miss (§11)."""
    n_a = int((pv.get("diag") or {}).get("tier_A", 0))
    if not gate["gated_metas"]:
        return ["M2_FILTER_MISS"] if n_a else ["NO_VALIDATED_PROPOSALS"]
    if not (pv.get("pool") or []):
        # validated action set empty: STOP is the ONLY legal action (§11) -- not a miss
        return ["NO_VALIDATED_PROPOSALS"]
    if terminal_reason in ("policy_stop", "stop_by_selector") and n_a > 0:
        return ["M3_SELECTION_MISS"]
    if acted_improvement is not None and acted_improvement <= 0 and n_a > 0:
        return ["M3_SELECTION_MISS"]
    if acted_improvement is not None and acted_improvement > 0:
        return ["M2_OK"]
    return ["UNKNOWN"]


def normal_m5_proposal_gate_r18(ast, metas, prop_feats, rolex, executor, pm, iid,
                                episode_id, step, sf, state_hash, ms_cur, cache=None):
    """§40 PERMANENT normal-M5 regression at the Proposal level: a dependency-
    completed (pair) Proposal whose REAL G_prop > 0 must be Proposal Tier-A
    directly -- NEVER routed through Memory (it is unconditionally retained)."""
    pv = proposal_validate_r18(ast, metas, prop_feats, rolex, executor, pm, iid,
                               episode_id, step, sf, state_hash, ms_cur, cache=cache)
    dep_pos = [r for r in pv["rows"] if r["kind"] == "pair" and r["g"] > 0.0]
    all_tier_a_ok = all(r["tier"] == "A" for r in dep_pos)
    return {"at_state": bool(pv["diag"]["full_pool_count"] > 0),
            "dep_pos": len(dep_pos), "all_tier_a_ok": all_tier_a_ok,
            "retained": [r["sig"] for r in dep_pos if r["tier"] == "A"],
            "pv_diag": dict(pv["diag"]), "n_full": pv["diag"]["full_pool_count"]}


def proposal_validation_state_stats(env, scorer, jpol, iid, progmem, ep=None,
                                    mem_budget=None):
    """R18 §33-39 per-state Proposal-validation diagnostic on the canonical S0
    states: runs the R14 adaptive root gate then REAL per-Proposal validation and
    returns the full event table (counts + top gains + validated pool, NO oracle).
    `ep` override for VAL states (env has no episode entry).  Pure stats; never
    feeds any runtime construction."""
    st = env["states"][iid]
    cache, executor = env["cache"], env["executor"]
    ep = int(env["ep_id_of"][iid]) if ep is None else int(ep)
    prop_feats, metas, agg = cache.proposals(st["problem"], st["schedule"], iid)
    if not metas:
        return {"skip": True, "iid": iid, "n_prop": 0}
    ast = cache.ast(st["problem"], st["schedule"], iid)
    ms = int(st["schedule"].makespan)
    base_h = schedule_hash(st["schedule"])
    sf = state_feature_vec(ms, ms, len(metas), agg["best_uhat"], agg["best_direct"],
                           agg["n_contrib"], agg["n_enab"])
    sf_t = torch.tensor(sf, dtype=torch.float32)
    rng = random.Random(0)
    gate = _m2_gate_step_r14(ast, metas, prop_feats, jpol.m2, executor, progmem,
                             iid, ep, 0, sf, rng)
    if not gate["gated_metas"]:
        return {"skip": True, "iid": iid, "n_prop": len(metas),
                "gated": 0, "m2_diag": dict(gate["diag"])}
    rolex = _rollex_of(ast, gate["gated_metas"], gate["gated_prop_feats"], sf_t)
    queries = [{"type": t2, "role": r_, "src": s_, "tgt": g_}
               for t2, r_, s_, g_ in zip(rolex["type"], rolex["role"],
                                         rolex["src"], rolex["tgt"])]
    mem = torch.tensor(progmem.features(iid, ep, 0, sf, queries), dtype=torch.float32)
    gmem = float(progmem.retrieval_gate(iid, ep, 0, sf))
    mem_sel = mem * gmem
    vs = build_validated_action_set_r18(
        ast, gate, rolex, scorer, executor, progmem, iid, ep, 0, sf, base_h, ms,
        mem_sel, cache={}, mem_budget=mem_budget)
    pv = vs["pval"]
    gains = sorted([r["g"] for r in pv["rows"] if r["valid"]], reverse=True)
    top_gains = [float(g) for g in gains[:10]]
    return {
        "iid": iid, "n_full": pv["diag"]["full_pool_count"],
        "n_positive": pv["diag"]["positive_count"],
        "n_zero": pv["diag"]["zero_count"], "n_negative": pv["diag"]["negative_count"],
        "n_infeasible": pv["diag"]["infeasible_count"],
        "n_memory_rescued": pv["diag"]["memory_rescued_count"],
        "n_validated": pv["diag"]["validated_count"],
        "n_tier_A": pv["diag"]["tier_A"], "n_tier_B": pv["diag"]["tier_B"],
        "validation_gains": top_gains, "cache_hits": pv["diag"]["cache_hits"],
        "cache_lookups": pv["diag"]["cache_lookups"],
        "m2_diag": dict(gate["diag"]),
        "m2_root_pos": int(gate["diag"].get("positive_probe_count", 0)),
        "root_has_pos": bool(gate["diag"].get("positive_probe_count", 0) > 0),
        "proposal_has_pos": bool(pv["diag"]["positive_count"] > 0),
        "skip": False,
    }


def r18_pool_distribution(groups):
    """§33-39/§51 pooled Proposal-validation event table across the collected
    validated trajectory states.  Aggregates the per-state pv_diag that every
    validated rec carries -> full pool size, positive density, tier counts, and
    validated-pool size spread plus the §51 compression ratio
    (mean validated  / mean Reasoner-full pool).  NO oracle anywhere."""
    d = {"states": 0, "full_counts": [], "pos_counts": [], "zero": 0, "neg": 0,
         "inf": 0, "tierA": 0, "tierB": 0, "tierC": 0, "val_sizes": [], "density": [],
         "probe_n": 0, "probe_ms": 0.0}
    for g in groups:
        d["probe_n"] += int(g.get("vprobe_n", 0))
        d["probe_ms"] += float(g.get("vprobe_ms", 0.0))
        for tr in g.get("trajs", []):
            for rec in tr.get("steps", []):
                pv = rec.get("pv_diag")
                if not pv:
                    continue
                d["states"] += 1
                d["full_counts"].append(int(pv["full_pool_count"]))
                d["pos_counts"].append(int(pv["positive_count"]))
                d["zero"] += int(pv["zero_count"])
                d["neg"] += int(pv["negative_count"])
                d["inf"] += int(pv["infeasible_count"])
                d["tierA"] += int(pv["tier_A"])
                d["tierB"] += int(pv["tier_B"])
                d["tierC"] += int(pv["tier_C"])
                d["val_sizes"].append(int(pv["validated_count"]))
                d["density"].append(float(pv["positive_density"]))
    n = len(d["val_sizes"])
    mean_full = float(np.mean(d["full_counts"])) if d["full_counts"] else 0.0
    mean_val = float(np.mean(d["val_sizes"])) if n else 0.0
    q = lambda v: (sorted(d["val_sizes"])[int(round(0.90 * (n - 1)))] if n else 0.0)  # noqa: E731
    return {
        "states": n, "probe_n": d["probe_n"], "probe_ms": d["probe_ms"],
        "mean_full": mean_full, "mean_positive": float(np.mean(d["pos_counts"])) if d["pos_counts"] else 0.0,
        "mean_density": float(np.mean(d["density"])) if d["density"] else 0.0,
        "zero_total": d["zero"], "neg_total": d["neg"], "inf_total": d["inf"],
        "tier_A_total": d["tierA"], "tier_B_total": d["tierB"], "tier_C_total": d["tierC"],
        "mean_validated": mean_val, "p90_validated": q(d["val_sizes"]),
        "n_val_gt16": sum(1 for x in d["val_sizes"] if x > 16),
        "n_val_gt32": sum(1 for x in d["val_sizes"] if x > 32),
        "n_val_gt48": sum(1 for x in d["val_sizes"] if x > 48),
        "n_val_gt64": sum(1 for x in d["val_sizes"] if x > 64),
        "compression_ratio": ((mean_val / mean_full) if mean_full > 0 else 1.0),
    }


def r18_rescue_precision(groups):
    """§46 MEMORY-RESCUE precision on collected validated groups: of the acted
    memory-rescued (Tier-B, G_prop<=0) Proposals, how many sat in a trajectory that
    still ended terminal-positive (the rescue paid off over the continuation).  The
    inverse is the FALSE-RESCUE rate and is gated against TO1_R18_FALSE_RESCUE_MAX
    (verdict E).  Tier-B is never eligible for terminal credit by construction don't
    mix with §4 observation semantics."""
    n_acted_b = 0
    n_b_terminal_ok = 0
    b_rows = []
    for g in groups:
        for tr in g.get("trajs", []):
            term_ok = float(tr.get("reward", 0.0)) > 0
            for rec in tr.get("steps", []):
                if rec.get("validation_tier") == "B":
                    n_acted_b += 1
                    if term_ok:
                        n_b_terminal_ok += 1
                    b_rows.append({"sig": rec.get("action_signature"),
                                   "g": float(rec.get("validation_gain", 0.0) or 0.0),
                                   "term_ok": term_ok})
    precision = (n_b_terminal_ok / n_acted_b) if n_acted_b else 0.0
    return {"n_acted_b": n_acted_b, "n_b_terminal_ok": n_b_terminal_ok,
            "rescue_precision": precision,
            "false_rescue_rate": (1.0 - precision), "rows": b_rows}


def r18_delayed_benefit(groups):
    """§44-45 DELAYED-BENEFIT diagnostic: the immediate validated (myopic) gate only
    ever acts proposals with REAL G_prop>0 (Tier-A) or memory-rescued (Tier-B).  If
    the gate is CORRECT, a trajectory that saw any Tier-A act must end terminal-
    positive.  delayed_benefit_rate = fraction of validated trajectories that ended
    at terminal 0 -- the share of rolled states where the myopic gate left the only
    real gains on the table.  Gated against TO1_R18_DELAYED_BENEFIT_MAX (verdict D)."""
    n_traj = 0
    n_term_ok = 0
    n_with_tier_a_act = 0
    n_a_term_ok = 0
    for g in groups:
        for tr in g.get("trajs", []):
            n_traj += 1
            term_ok = float(tr.get("reward", 0.0)) > 0
            if term_ok:
                n_term_ok += 1
            has_a = any(rec.get("validation_tier") == "A" for rec in tr.get("steps", []))
            if has_a:
                n_with_tier_a_act += 1
                if term_ok:
                    n_a_term_ok += 1
    rate = (1.0 - n_term_ok / n_traj) if n_traj else 0.0
    return {"n_traj": n_traj, "n_terminal_ok": n_term_ok,
            "delayed_benefit_rate": rate,
            "n_with_tier_a_act": n_with_tier_a_act,
            "tier_a_realized": (n_a_term_ok / n_with_tier_a_act) if n_with_tier_a_act else 0.0}
# __R18_CHUNK_1__
