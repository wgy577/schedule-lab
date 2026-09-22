#!/usr/bin/env python3
"""T1-B5.2-INTERACTION-AWARE-PAIR-RANKING-R1.

Resolves the single problem left by T1-BOUNDED-2EDIT-PROPOSAL-COMPOSER (verdict C):
the bounded 2-edit pool already covers the oracle-best (16/17), but the *additive*
scorer ranks genuinely-good super-additive joints (contributor + enabler) to the
back, because the enabler carries ~0 B5 score and the pair's value is the
*interaction*, not the sum of two singles.

This round does three things at once, all deterministic / bounded / non-RL:

  1. CANDIDATE EXPANSION  -- add a "Source B" enabler channel on top of the
     Source-A contributor channel (Top-K contributors).  For every contributor
     edit X: M_high -> M_mid, the ops Y currently on M_mid that can relocate
     away (Y: M_mid -> M_low) are pulled in as ENABLER candidates, even when
     b5_score(Y) ~ 0 and M_mid has no active Appearance.  Roles are annotated
     CONTRIBUTOR / ENABLER; pair provenance keeps dependency_trace.

  2. B5.2 INTERACTION HEAD -- a tiny MLP over [h_c[u], h_c[v], pair_struct]
     predicting the frozen teacher interaction I_uv.  B5.1 encoder / c prior /
     e propagation are all FROZEN (no grad).  Nothing else trains.

  3. SCIENTIFIC CHECK + COMPARATIVE RANKING -- (a) teacher I_uv vs runtime I_ms
     alignment; (b) additive vs interaction-aware pair ranking over the SAME
     expanded pool, evaluated on runtime makespan (Frozen-Local, diagnostic).

identified=false, formal_test_access=0, no RL, I_uv pilot-branch only.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "outputs" / "b5_2_interaction_ranking"
ORACLE_DIR = ROOT / "outputs" / "joint_oracle_audit"
TEACHER_DIR = ROOT / "outputs" / "b5_route2_teacher"
DEFAULT_CKPT = ROOT / "outputs" / "b5_1_pilot_A1_v1_1_adapter" / "b5_1_shared.pt"

K = 3                 # canonical B5 Top-K contributors per important block
L = 2                 # top-L legal ROUTE edits per op (contributor AND enabler)
W_IMPACT = 0.1        # deterministic processing-time-saving prior weight
LAMBDA = 1.0          # interaction correction weight (fixed first version)
PAIR_BUDGETS = (1, 3, 5, 10, 20)
PAIR_FEAT_DIM = 16
HIDDEN_DIM = 128
B5_EPS = 1e-3  # numerical-zero threshold: only genuinely positive B5 scores count as contributors


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


runmod = _load("run_v5_mainline_loop", ROOT / "scripts" / "run_v5_mainline_loop.py")
pilot = _load("b5_1_train_pilot", ROOT / "scripts" / "b5_1_train_pilot.py")

from causal_schedule_lab.intervention import ScheduleGraphView  # noqa: E402
from causal_schedule_lab.intervention.composite_legality import (  # noqa: E402
    check_composite_structural_legality,
)
from causal_schedule_lab.legal_edit_enumerator import LegalEditEnumerator  # noqa: E402
from causal_schedule_lab.m2_v5_schema_v1 import EDIT_ROUTE, CausalInterventionProposal  # noqa: E402
from causal_schedule_lab.teacher.atomic_counterfactual_executor import (  # noqa: E402
    AtomicCounterfactualExecutor,
)
from causal_schedule_lab.validation import schedule_hash  # noqa: E402


# ---------------------------------------------------------------------------
# deterministic pair relation features
# ---------------------------------------------------------------------------
def _mode_min_dur(problem, op_id: str, machine: str):
    op = problem.operation_map()[op_id]
    ds = [m.duration for m in op.modes if m.resources and m.resources[0] == machine]
    return min(ds) if ds else None


def _cur_dur(problem, schedule, op_id: str):
    asg = schedule.assignment_map()[op_id]
    return float(problem.mode_map()[asg.mode_id][1].duration)


def _rel_saving_op(problem, schedule, op_id: str, target_machine: str) -> float:
    p_cur = _cur_dur(problem, schedule, op_id)
    p_tgt = _mode_min_dur(problem, op_id, target_machine)
    if p_tgt is None:
        return 0.0
    return max(0.0, (p_cur - p_tgt) / max(p_cur, 1.0))


def pair_feature_vec(problem, schedule, enum, op_u, src_u, tgt_u, op_v, src_v, tgt_v):
    """16-dim deterministic structural feature vector for a pair of ROUTE edits."""
    ms = max(enum.makespan, 1.0)
    loads = enum.loads
    opmap = problem.operation_map()
    same_job = 1.0 if opmap[op_u].job_id == opmap[op_v].job_id else 0.0
    same_source = 1.0 if src_u == src_v else 0.0
    same_target = 1.0 if tgt_u == tgt_v else 0.0
    u_tgt_eq_v_src = 1.0 if tgt_u == src_v else 0.0
    v_tgt_eq_u_src = 1.0 if tgt_v == src_u else 0.0
    swap_like = 1.0 if (src_u == tgt_v and tgt_u == src_v) else 0.0
    shared = 1.0 if ({src_u, tgt_u} & {src_v, tgt_v}) else 0.0
    independent = 1.0 - shared
    rel_saving_u = _rel_saving_op(problem, schedule, op_u, tgt_u)
    rel_saving_v = _rel_saving_op(problem, schedule, op_v, tgt_v)
    p_cur_u = _cur_dur(problem, schedule, op_u)
    p_cur_v = _cur_dur(problem, schedule, op_v)
    p_tgt_u = _mode_min_dur(problem, op_u, tgt_u) or p_cur_u
    p_tgt_v = _mode_min_dur(problem, op_v, tgt_v) or p_cur_v
    return [
        same_source, same_target, u_tgt_eq_v_src, v_tgt_eq_u_src,
        swap_like, shared, independent, same_job,
        rel_saving_u, rel_saving_v,
        loads.get(src_u, 0.0) / ms, loads.get(tgt_u, 0.0) / ms,
        loads.get(src_v, 0.0) / ms, loads.get(tgt_v, 0.0) / ms,
        (p_tgt_u - p_cur_u) / ms, (p_tgt_v - p_cur_v) / ms,
    ]


def parse_route_atom(atom: str):
    """ROUTE:op:src->tgt  ->  (op, src, tgt);  None otherwise."""
    if not atom.startswith("ROUTE:"):
        return None
    parts = atom.split(":")
    if len(parts) < 3:
        return None
    op = parts[1]
    try:
        src, tgt = parts[2].split("->")
    except ValueError:
        return None
    return op, src, tgt


# ---------------------------------------------------------------------------
# B5.2 interaction head (the ONLY trained parameters)
# ---------------------------------------------------------------------------
class PairInteractionHead(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM, feat_dim: int = PAIR_FEAT_DIM):
        super().__init__()
        in_dim = 2 * hidden_dim + feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_u, h_v, feat):
        return self.net(torch.cat([h_u, h_v, feat], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# candidate pool: contributor (Source A) + enabler (Source B)
# ---------------------------------------------------------------------------
def _score_atomic(e, op_b5: dict[str, float]) -> float:
    f = dict(e.features)
    pc = float(f.get("p_current", 0.0) or 0.0)
    pt = float(f.get("p_target", 0.0) or 0.0)
    return op_b5.get(e.operation_id, 0.0) + W_IMPACT * max(0.0, (pc - pt) / max(pc, 1.0))


def _sig(e) -> str:
    return f"{e.edit_type}:{e.operation_id}:{e.source_machine}->{e.target_machine}"


def build_pool(problem, schedule, op_b5: dict[str, float]):
    """Return (contributor_edits, enabler_edits, role_by_sig)."""
    all_ops = tuple(sorted(schedule.assignment_map()))
    enum = LegalEditEnumerator(problem, schedule)
    edits = tuple(enum.enumerate(
        all_ops, request_route=True, request_seq_swap=False,
        request_seq_insert=False, request_timing_shift=False,
    ))
    route = [e for e in edits if e.edit_type == EDIT_ROUTE]

    # Source A -- contributors (B5 Top-K ops)
    by_op = collections.defaultdict(list)
    for e in route:
        if e.operation_id in op_b5:
            by_op[e.operation_id].append(e)
    contributor_edits = []
    for op, es in by_op.items():
        es_sorted = sorted(es, key=lambda e: _score_atomic(e, op_b5), reverse=True)[:L]
        contributor_edits.extend(es_sorted)

    contrib_tgt_machines = {e.target_machine for e in contributor_edits}
    contributor_ops = set(op_b5.keys())

    # Source B -- enablers (ops on a contributor target machine, NOT contributors).
    # An enabler's value is *releasing the contributor's target machine*, not its
    # own rel_saving, so we keep EVERY legal relocation away from that machine
    # (naturally bounded by #machines per op) -- NOT a top-L rel_saving cap.
    enabler_edits = []
    seen_enabler = set()
    for e in route:
        if e.operation_id in contributor_ops:
            continue
        if e.source_machine not in contrib_tgt_machines:
            continue
        if e.target_machine == e.source_machine:
            continue
        sig = _sig(e)
        if sig in seen_enabler:
            continue
        seen_enabler.add(sig)
        enabler_edits.append(e)

    role_by_sig = {_sig(e): "CONTRIBUTOR" for e in contributor_edits}
    role_by_sig.update({_sig(e): "ENABLER" for e in enabler_edits})
    return enum, contributor_edits, enabler_edits, role_by_sig


def _classify_pair(u, v) -> str:
    us, ut = u.source_machine, u.target_machine
    vs, vt = v.source_machine, v.target_machine
    if us == vt and ut == vs:
        return "SWAP_LIKE"
    if us == vt or vs == ut:
        return "ENABLING"
    if len({us, ut} & {vs, vt}) > 0:
        return "SHARED_RESOURCE"
    return "INDEPENDENT"


def _exec_edits(executor, problem, schedule, edits, base_ms, base_hash):
    prop = CausalInterventionProposal(
        proposal_id="b52", appearance_id="b52",
        edits=tuple(edits), root_path=tuple(e.operation_id for e in edits),
    )
    atoms = runmod._d6_proposal_to_atoms(problem, schedule, prop)
    if atoms is None:
        return {"feasible": False, "delta": None, "improvement": None, "reason": "unmappable"}
    if schedule_hash(schedule) != base_hash:
        raise SystemExit("BLOCKER: executor mutated baseline")
    res = runmod._d6_execute(executor, problem, schedule, atoms)
    if not res.feasible or res.schedule is None:
        return {"feasible": False, "delta": None, "improvement": None, "reason": res.report.reason}
    succ_ms = int(res.schedule.makespan)
    return {"feasible": True, "delta": succ_ms - base_ms, "improvement": base_ms - succ_ms,
            "reason": "ok"}


# ---------------------------------------------------------------------------
# teacher I_uv vs runtime I_ms alignment (STEP 6)
# ---------------------------------------------------------------------------
def alignment_check():
    ms_single = {}
    for split in ("train", "val"):
        for l in (TEACHER_DIR / f"single_{split}.jsonl").open():
            r = json.loads(l)
            if r["status"] == "VALID_ANCHORED" and r.get("makespan_delta") is not None:
                ms_single[(r["instance_id"], r["appearance_anchor_id"], r["candidate_operation"])] = r["makespan_delta"]
    rows = []
    for split in ("train", "val"):
        for l in (TEACHER_DIR / f"pair_{split}.jsonl").open():
            r = json.loads(l)
            if r["status"] != "VALID_ANCHORED" or r.get("I_uv") is None:
                continue
            if r.get("joint_makespan_delta") is None:
                continue
            du = ms_single.get((r["instance_id"], r["appearance_anchor_id"], r["u"]))
            dv = ms_single.get((r["instance_id"], r["appearance_anchor_id"], r["v"]))
            if du is None or dv is None:
                continue
            I_uv = r["I_uv"]
            I_ms = r["joint_makespan_delta"] - du - dv
            rows.append((r["split"], I_uv, I_ms))
    out = {}
    for split in ("train", "val", "all"):
        sub = rows if split == "all" else [x for x in rows if x[0] == split]
        if not sub:
            continue
        iuv = np.array([x[1] for x in sub]); ims = np.array([x[2] for x in sub])
        ra = np.argsort(np.argsort(iuv)); rb = np.argsort(np.argsort(ims))
        n = len(sub); k = max(1, n // 5)
        top_ms = set(np.argsort(-np.abs(ims))[:k]); top_uv = set(np.argsort(-np.abs(iuv))[:k])
        out[split] = {
            "n": n,
            "sign_agreement": float(np.mean(np.sign(iuv) == np.sign(ims))),
            "pearson": float(np.corrcoef(iuv, ims)[0, 1]),
            "spearman": float(np.corrcoef(ra, rb)[0, 1]),
            f"large_interaction_retrieval_top{k}": float(len(top_ms & top_uv) / k),
        }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    insts, _ = pilot._instance_paths()
    order = ([i for i in insts.values() if i["split"] == "train"]
             + [i for i in insts.values() if i["split"] == "val"])

    print("[build] reconstructing states ...", flush=True)
    states = {i["instance_id"]: pilot.build_state(i) for i in order}

    important = {}
    for split in ("train", "val"):
        groups, _ = pilot.load_teacher_groups(split, states)
        for iid, gs in groups.items():
            important.setdefault(iid, set()).update(g["appearance_id"] for g in gs)
    for iid in states:
        important.setdefault(iid, set())

    first_iid = order[0]["instance_id"]
    model = pilot.from_manifest_b5(states[first_iid]["bundle"], b5_adapter=True, b5_mod_dim=16)
    model.load_state_dict(torch.load(Path(args.ckpt), map_location="cpu"))
    model.eval()

    # ---- frozen node embeddings h_c + per-state B5 top-K scores -------------
    emb: dict[str, torch.Tensor] = {}   # iid -> h_c [N,H]
    op_b5_by_state: dict[str, dict[str, float]] = {}
    print("[embed] computing frozen h_c + B5 scores ...", flush=True)
    for inst in order:
        iid = inst["instance_id"]
        st = states[iid]
        model.bind_runtime_context(st["context"], node_ids=list(st["node_ids"]))
        with torch.no_grad():
            out = model.attribution_forward(st["batch"], st["c_prior"])
            h_a, h_c, _, _ = model._encode_h_a(st["batch"])
        emb[iid] = h_c
        scores = out.per_block_candidate_score.numpy()
        bids = [b for b in st["block_ids"] if b in important[iid]]
        op_b5: dict[str, float] = {}
        for bi in range(len(bids)):
            bi_abs = st["block_index"][bids[bi]]
            for ni in np.argsort(-scores[bi_abs]):
                s = float(scores[bi_abs, ni])
                if s <= B5_EPS:
                    break
                nid = st["node_ids"][ni]
                if nid.startswith("operation:"):
                    op = nid[len("operation:"):]
                    op_b5[op] = max(op_b5.get(op, 0.0), s)
        op_b5_by_state[iid] = op_b5

    # ---- STEP 6: teacher I_uv vs runtime I_ms --------------------------------
    alignment = alignment_check()
    print("\n[STEP 6] teacher I_uv vs runtime I_ms alignment:", json.dumps(alignment, indent=2))

    # ---- training set: routing+routing teacher pairs with I_uv --------------
    train_X, train_y = [], []
    val_X, val_y = [], []
    n_skip = 0
    for split in ("train", "val"):
        for l in (TEACHER_DIR / f"pair_{split}.jsonl").open():
            r = json.loads(l)
            if r.get("I_uv") is None or r["status"] != "VALID_ANCHORED":
                continue
            if r.get("edit_kind_u") != "routing" or r.get("edit_kind_v") != "routing":
                continue
            pu = parse_route_atom(r["atom_u"]); pv = parse_route_atom(r["atom_v"])
            if pu is None or pv is None:
                continue
            iid = r["instance_id"]
            if iid not in states:
                n_skip += 1
                continue
            st = states[iid]
            h_c = emb[iid]
            ni_u = st["node_index"].get(f"operation:{pu[0]}")
            ni_v = st["node_index"].get(f"operation:{pv[0]}")
            if ni_u is None or ni_v is None:
                n_skip += 1
                continue
            enum = LegalEditEnumerator(st["problem"], st["schedule"])
            feat = pair_feature_vec(st["problem"], st["schedule"], enum,
                                    pu[0], pu[1], pu[2], pv[0], pv[1], pv[2])
            hu = h_c[ni_u]; hv = h_c[ni_v]
            X = torch.cat([hu, hv, torch.tensor(feat, dtype=torch.float32)])
            y = float(r["I_uv"])
            if split == "train":
                train_X.append(X); train_y.append(y)
            else:
                val_X.append(X); val_y.append(y)
    train_X = torch.stack(train_X); train_y = torch.tensor(train_y, dtype=torch.float32)
    val_X = torch.stack(val_X); val_y = torch.tensor(val_y, dtype=torch.float32)
    print(f"[data] interaction head training: TRAIN={train_X.shape[0]} VAL={val_X.shape[0]} skip={n_skip}")

    # ---- train interaction head (predict teacher I_uv) -----------------------
    head = PairInteractionHead()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    mse = nn.MSELoss()
    steps = args.max_steps
    for step in range(steps):
        head.train()
        opt.zero_grad()
        idx = torch.randperm(train_X.shape[0])[:args.batch_size]
        pred = head(train_X[idx, :HIDDEN_DIM], train_X[idx, HIDDEN_DIM:2 * HIDDEN_DIM],
                    train_X[idx, 2 * HIDDEN_DIM:])
        loss = mse(pred, train_y[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
    head.eval()
    with torch.no_grad():
        pred_tr = head(train_X[:, :HIDDEN_DIM], train_X[:, HIDDEN_DIM:2 * HIDDEN_DIM], train_X[:, 2 * HIDDEN_DIM:])
        pred_va = head(val_X[:, :HIDDEN_DIM], val_X[:, HIDDEN_DIM:2 * HIDDEN_DIM], val_X[:, 2 * HIDDEN_DIM:])
    def _corr(a, b):
        a = a.numpy(); b = b.numpy()
        return float(np.corrcoef(a, b)[0, 1])
    head_metrics = {
        "train_mse": float(mse(pred_tr, train_y).item()),
        "val_mse": float(mse(pred_va, val_y).item()),
        "train_pearson": _corr(pred_tr, train_y),
        "val_pearson": _corr(pred_va, val_y),
    }
    print(f"[B5.2 head] I_uv prediction: {head_metrics}")

    # ---- build expanded pool + eval per state --------------------------------
    executor = AtomicCounterfactualExecutor()
    oracle = json.loads(ORACLE_DIR.joinpath("result.json").read_text())
    oracle_state = {s["state_id"]: s for s in oracle["per_state"]}

    def oracle_best_of(s):
        rows = [r for r in s["joint_rows"] if r["improvement"] is not None]
        return max(rows, key=lambda r: r["improvement"]) if rows else None

    states_rows = []
    for inst in order:
        iid = inst["instance_id"]
        st = states[iid]
        op_b5 = op_b5_by_state[iid]
        problem, schedule = st["problem"], st["schedule"]
        base_ms = int(schedule.makespan)
        base_hash = schedule_hash(schedule)
        h_c = emb[iid]
        gv = ScheduleGraphView.from_problem_schedule(problem, schedule)

        enum, contrib, enab, role = build_pool(problem, schedule, op_b5)
        n_contrib = len(contrib); n_enab = len(enab)

        # pairs (dedup by sig); CC / CE / EE
        pool = contrib + enab
        sig_edit = {_sig(e): e for e in pool}
        pairs = []
        type_counts = {"CC": 0, "CE": 0, "EE": 0}
        pair_relation = collections.Counter()
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                u, v = pool[i], pool[j]
                if u.operation_id == v.operation_id:
                    continue
                leg = check_composite_structural_legality(gv, (u, v))
                if not leg.legal:
                    continue
                ru, rv = role[_sig(u)], role[_sig(v)]
                rr = "CC" if (ru == "CONTRIBUTOR" and rv == "CONTRIBUTOR") else \
                     "EE" if (ru == "ENABLER" and rv == "ENABLER") else "CE"
                type_counts[rr] += 1
                pair_relation[_classify_pair(u, v)] += 1
                # additive + interaction features
                add = _score_atomic(u, op_b5) + _score_atomic(v, op_b5)
                ni_u = st["node_index"][f"operation:{u.operation_id}"]
                ni_v = st["node_index"][f"operation:{v.operation_id}"]
                feat = pair_feature_vec(problem, schedule, enum,
                                        u.operation_id, u.source_machine, u.target_machine,
                                        v.operation_id, v.source_machine, v.target_machine)
                pairs.append({
                    "u": u, "v": v, "usig": _sig(u), "vsig": _sig(v),
                    "add": add, "feat": torch.tensor(feat, dtype=torch.float32),
                    "type": _classify_pair(u, v), "rr": rr,
                    "ni_u": ni_u, "ni_v": ni_v, "ru": ru, "rv": rv,
                })

        # interaction-aware score via trained head
        for p in pairs:
            hu = h_c[p["ni_u"]]; hv = h_c[p["ni_v"]]
            with torch.no_grad():
                ihat = float(head(hu.unsqueeze(0), hv.unsqueeze(0), p["feat"].unsqueeze(0)).item())
            p["ihat"] = ihat
            p["score_ia"] = p["add"] + LAMBDA * ihat

        pairs.sort(key=lambda p: -p["add"])
        pairs_ia = sorted(pairs, key=lambda p: -p["score_ia"])

        # single baseline
        best_single = None
        for e in pool:
            r = _exec_edits(executor, problem, schedule, (e,), base_ms, base_hash)
            if r["improvement"] is not None and (best_single is None or r["improvement"] > best_single):
                best_single = r["improvement"]

        # oracle-best (Source-A oracle) rank under additive vs interaction-aware
        ob = oracle_best_of(oracle_state.get(iid, {}))
        ob_rank_add = ob_rank_ia = None
        ob_in_pool = None
        if ob is not None:
            ue = sig_edit.get(ob["u"]); ve = sig_edit.get(ob["v"])
            if ue is not None and ve is not None and ue.operation_id != ve.operation_id:
                ob_in_pool = True
                ob_add = _score_atomic(ue, op_b5) + _score_atomic(ve, op_b5)
                ob_rank_add = 1 + sum(1 for p in pairs if p["add"] > ob_add + 1e-12)
                # interaction-aware rank
                ni_u = st["node_index"][f"operation:{ue.operation_id}"]
                ni_v = st["node_index"][f"operation:{ve.operation_id}"]
                feat = pair_feature_vec(problem, schedule, enum,
                                        ue.operation_id, ue.source_machine, ue.target_machine,
                                        ve.operation_id, ve.source_machine, ve.target_machine)
                hu = h_c[ni_u]; hv = h_c[ni_v]
                with torch.no_grad():
                    ihat_ob = float(head(hu.unsqueeze(0), hv.unsqueeze(0), torch.tensor(feat).unsqueeze(0)).item())
                ob_score_ia = ob_add + LAMBDA * ihat_ob
                ob_rank_ia = 1 + sum(1 for p in pairs if p["score_ia"] > ob_score_ia + 1e-12)
            else:
                ob_in_pool = False

        # per-budget Frozen-Local eval (additive vs interaction-aware)
        def budget_eval(pairs_sorted, max_n):
            evs = []
            for p in pairs_sorted[:max_n]:
                r = _exec_edits(executor, problem, schedule, (p["u"], p["v"]), base_ms, base_hash)
                evs.append({"usig": p["usig"], "vsig": p["vsig"], "improvement": r["improvement"]})
            out = {}
            for N in PAIR_BUDGETS:
                imp = [e["improvement"] for e in evs[:N] if e["improvement"] is not None]
                out[N] = {"best": max(imp) if imp else None,
                          "n_improving": sum(1 for x in imp if x > 0),
                          "beats_single": (max(imp) if imp else -1) > (best_single or -1) if imp else False}
            return out, evs

        bud_add, evs_add = budget_eval(pairs, max(PAIR_BUDGETS))
        bud_ia, evs_ia = budget_eval(pairs_ia, max(PAIR_BUDGETS))

        states_rows.append({
            "state_id": iid, "split": inst["split"], "base_makespan": base_ms,
            "n_contrib": n_contrib, "n_enab": n_enab,
            "pair_type_counts": type_counts, "pair_relation": dict(pair_relation),
            "n_pairs": len(pairs), "best_single": best_single,
            "oracle_best": ob, "oracle_best_in_pool": ob_in_pool,
            "oracle_best_rank_add": ob_rank_add, "oracle_best_rank_ia": ob_rank_ia,
            "budget_add": bud_add, "budget_ia": bud_ia,
        })
        print(f"[state] {iid:26s} contrib={n_contrib} enab={n_enab} pairs={len(pairs)} "
              f"ob_in_pool={ob_in_pool} rank_add={ob_rank_add} rank_ia={ob_rank_ia} "
              f"best_single={best_single}", flush=True)

    # ---- aggregate -----------------------------------------------------------
    def _med(xs):
        return float(np.median(xs)) if xs else None

    agg = {
        "K": K, "L": L, "lambda": LAMBDA, "pair_feat_dim": PAIR_FEAT_DIM,
        "alignment": alignment,
        "head_metrics": head_metrics,
        "n_states": len(states_rows),
        "n_contrib_total": sum(r["n_contrib"] for r in states_rows),
        "n_enab_total": sum(r["n_enab"] for r in states_rows),
        "pair_types_total": {k: sum(r["pair_type_counts"][k] for r in states_rows)
                             for k in ("CC", "CE", "EE")},
        "pair_relation_total": collections.Counter(),
        "oracle_best_in_pool": sum(1 for r in states_rows if r["oracle_best_in_pool"]),
        "oracle_best_rank_add_median": _med([r["oracle_best_rank_add"] for r in states_rows if r["oracle_best_rank_add"] is not None]),
        "oracle_best_rank_ia_median": _med([r["oracle_best_rank_ia"] for r in states_rows if r["oracle_best_rank_ia"] is not None]),
    }
    for r in states_rows:
        for k, v in r["pair_relation"].items():
            agg["pair_relation_total"][k] += v
    agg["pair_relation_total"] = dict(agg["pair_relation_total"])

    # P@K (additive vs interaction-aware) over in-pool oracle-best states
    for Kk in (1, 3, 5, 10):
        rankable = [r for r in states_rows if r["oracle_best_rank_add"] is not None]
        agg[f"P@{Kk}_add"] = sum(1 for r in rankable if r["oracle_best_rank_add"] <= Kk) / len(rankable) if rankable else None
        rankable = [r for r in states_rows if r["oracle_best_rank_ia"] is not None]
        agg[f"P@{Kk}_ia"] = sum(1 for r in rankable if r["oracle_best_rank_ia"] <= Kk) / len(rankable) if rankable else None

    # strong-interaction median rank (|I_ms|>=10), add vs ia
    strong_add = [r["oracle_best_rank_add"] for r in states_rows
                  if r["oracle_best"] is not None and r["oracle_best_rank_add"] is not None
                  and abs(r["oracle_best"]["I_ms"]) >= 10]
    strong_ia = [r["oracle_best_rank_ia"] for r in states_rows
                 if r["oracle_best"] is not None and r["oracle_best_rank_ia"] is not None
                 and abs(r["oracle_best"]["I_ms"]) >= 10]
    agg["strong_interaction_median_rank_add"] = _med(strong_add)
    agg["strong_interaction_median_rank_ia"] = _med(strong_ia)

    # budget sweep aggregates
    agg["budgets"] = {}
    for N in PAIR_BUDGETS:
        agg["budgets"][N] = {
            "add_positive_joint_states": sum(1 for r in states_rows if r["budget_add"][N]["n_improving"] > 0),
            "ia_positive_joint_states": sum(1 for r in states_rows if r["budget_ia"][N]["n_improving"] > 0),
            "add_beats_single_states": sum(1 for r in states_rows if r["budget_add"][N]["beats_single"]),
            "ia_beats_single_states": sum(1 for r in states_rows if r["budget_ia"][N]["beats_single"]),
            "add_best_joint": max((r["budget_add"][N]["best"] for r in states_rows if r["budget_add"][N]["best"] is not None), default=None),
            "ia_best_joint": max((r["budget_ia"][N]["best"] for r in states_rows if r["budget_ia"][N]["best"] is not None), default=None),
        }

    (OUT_DIR / "result.json").write_text(json.dumps(agg, indent=2, default=str))
    for r in states_rows:
        (OUT_DIR / f"state_{r['state_id']}.json").write_text(json.dumps(r, indent=2, default=str))

    print("\n==================== B5.2 INTERACTION-AWARE RANKING ====================")
    print(f"  candidate: contrib={agg['n_contrib_total']} enab={agg['n_enab_total']} "
          f"pair_types={agg['pair_types_total']}")
    print(f"  oracle-best in pool: {agg['oracle_best_in_pool']}/{agg['n_states']}")
    print(f"  oracle-best rank median: add={agg['oracle_best_rank_add_median']} ia={agg['oracle_best_rank_ia_median']}")
    print(f"  strong-interaction median rank: add={agg['strong_interaction_median_rank_add']} ia={agg['strong_interaction_median_rank_ia']}")
    for Kk in (1, 3, 5, 10):
        print(f"  P@{Kk}: add={agg[f'P@{Kk}_add']} ia={agg[f'P@{Kk}_ia']}")
    for N in PAIR_BUDGETS:
        b = agg["budgets"][N]
        print(f"  budget={N:2d}: add_pos={b['add_positive_joint_states']} ia_pos={b['ia_positive_joint_states']} "
              f"add_beat={b['add_beats_single_states']} ia_beat={b['ia_beats_single_states']} "
              f"add_best={b['add_best_joint']} ia_best={b['ia_best_joint']}")
    print(f"  Saved: {OUT_DIR / 'result.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
