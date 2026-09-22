#!/usr/bin/env python3
"""T1-M3-MAKESPAN-UTILITY-AND-PAIR-RANKING-R1.

Formally move from B5 attribution (appearance-side) to M3 proposal utility
(makespan-side).  The previous round proved teacher appearance interaction
I_uv is ORTHOGONAL to runtime makespan interaction I_ms (pearson 0.002), so
makespan-side proposal utility must be learned independently by M3.

This round trains three small heads on FROZEN B5.1 h_c + deterministic features,
with labels generated exclusively by Frozen-Local execution (sibling baseline S_t):

  - M3SingleUtilityHead :  U_hat(e)  = Cmax(S_t) - Cmax(S_t after e)
  - M3PairResidualHead  :  I_hat_ms(u,v) = U(u,v) - U(u) - U(v)  (residual)
  - DirectPairUtilityHead: U_hat_pair(u,v)  (diagnostic baseline, no decomposition)

  canonical M3 pair score = U_hat(u) + U_hat(v) + I_hat_ms(u,v)

Unified sign convention (project-wide this round):
  U_ms(p) = Cmax(base) - Cmax(after)     > 0 = improvement, 0 = neutral, < 0 = worse.

identified=false, formal_test_access=0, Formal TEST SEALED, no RL.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "outputs" / "m3_makespan_utility"
ORACLE_DIR = ROOT / "outputs" / "joint_oracle_audit"
TEACHER_DIR = ROOT / "outputs" / "b5_route2_teacher"
B52_DIR = ROOT / "outputs" / "b5_2_interaction_ranking"
DEFAULT_CKPT = ROOT / "outputs" / "b5_1_pilot_A1_v1_1_adapter" / "b5_1_shared.pt"

K = 3          # B5 Top-K contributors per important block
L = 2          # top-L legal ROUTE edits per op (contributor)
W_IMPACT = 0.1
LAMBDA = 1.0   # residual correction weight (fixed)
PAIR_BUDGETS = (1, 3, 5, 10, 20)
PAIR_FEAT_DIM = 16
SINGLE_FEAT_DIM = 12
HIDDEN_DIM = 128
B5_EPS = 1e-3

# pair-label sampling budget (per state, per relation type)
PAIR_SAMPLE_BUDGET = {"ENABLING": 250, "SHARED_RESOURCE": 100,
                      "SWAP_LIKE": 100, "INDEPENDENT": 100}

# method name -> scored-entry key
SCORE_KEY = {"add": "add", "b52": "b52s", "m3": "m3s", "direct": "directs"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


runmod = _load("run_v5_mainline_loop", ROOT / "scripts" / "run_v5_mainline_loop.py")
pilot = _load("b5_1_train_pilot", ROOT / "scripts" / "b5_1_train_pilot.py")
b52 = _load("t1_b5_2_interaction_ranking", ROOT / "scripts" / "t1_b5_2_interaction_ranking.py")

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
# deterministic single-edit feature vector
# ---------------------------------------------------------------------------
def single_feature_vec(e, op_b5: dict[str, float], role: str) -> list[float]:
    f = dict(e.features)
    pc = float(f.get("p_current", 0.0) or 0.0)
    pt = float(f.get("p_target", 0.0) or 0.0)
    dp = float(f.get("delta_p", 0.0) or 0.0)
    sl = float(f.get("source_load", 0.0) or 0.0)
    tl = float(f.get("target_load", 0.0) or 0.0)
    srl = float(f.get("source_relative_load", 0.0) or 0.0)
    trl = float(f.get("target_relative_load", 0.0) or 0.0)
    flex = float(f.get("flexibility", 0.0) or 0.0)
    rc = float(f.get("receiver_context", 0.0) or 0.0)
    rel_saving = max(0.0, (pc - pt) / max(pc, 1.0))
    b5s = op_b5.get(e.operation_id, 0.0)
    role_num = 1.0 if role == "ENABLER" else 0.0
    return [rel_saving, dp, pc, pt, sl, tl, srl, trl, flex, rc, b5s, role_num]


# ---------------------------------------------------------------------------
# trainable heads
# ---------------------------------------------------------------------------
class M3SingleUtilityHead(nn.Module):
    """Input X = [h (128), single_feat (12)] concatenated."""

    def __init__(self, feat_dim: int = SINGLE_FEAT_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(hidden + feat_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, X):
        return self.net(X).squeeze(-1)


class M3PairResidualHead(nn.Module):
    """Input X = [h_u, h_v, feat_u, feat_v, pair_struct(16), U_hat_u, U_hat_v]."""

    def __init__(self, feat_dim: int = SINGLE_FEAT_DIM, pair_dim: int = PAIR_FEAT_DIM,
                 hidden: int = HIDDEN_DIM):
        super().__init__()
        self.hidden = hidden
        self.feat_dim = feat_dim
        self.pair_dim = pair_dim
        in_dim = 2 * hidden + 2 * feat_dim + pair_dim + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, X):
        return self.net(X).squeeze(-1)


class DirectPairUtilityHead(nn.Module):
    """Input X = [h_u, h_v, feat_u, feat_v, pair_struct(16)]."""

    def __init__(self, feat_dim: int = SINGLE_FEAT_DIM, pair_dim: int = PAIR_FEAT_DIM,
                 hidden: int = HIDDEN_DIM):
        super().__init__()
        self.hidden = hidden
        self.feat_dim = feat_dim
        self.pair_dim = pair_dim
        in_dim = 2 * hidden + 2 * feat_dim + pair_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, X):
        return self.net(X).squeeze(-1)


# ---------------------------------------------------------------------------
# label generation (Frozen-Local)
# ---------------------------------------------------------------------------
def _exec(executor, problem, schedule, edits, base_ms, base_hash):
    return b52._exec_edits(executor, problem, schedule, tuple(edits), base_ms, base_hash)


def _within_state_rank_loss(pred, y, groups, margin=0.0, n_pairs=200):
    """Pairwise margin ranking loss over sampled within-group pairs."""
    if pred.shape[0] < 2:
        return torch.zeros((), device=pred.device)
    loss = torch.zeros((), device=pred.device)
    cnt = 0
    for gid in torch.unique(groups):
        idx = (groups == gid).nonzero(as_tuple=False).squeeze(-1)
        if idx.shape[0] < 2:
            continue
        for _ in range(min(n_pairs, idx.shape[0] * 2)):
            i, j = idx[torch.randint(idx.shape[0], (2,), device=pred.device)]
            if i == j:
                continue
            if (y[i] - y[j]).abs() < 1e-9:
                continue
            sgn = (y[i] - y[j]).sign()
            loss = loss + torch.clamp(margin - sgn * (pred[i] - pred[j]), min=0.0)
            cnt += 1
    return loss / max(cnt, 1)


def _train_head(model, X, y, groups, epochs, lr, batch, alpha_rank=0.1, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()
    n = X[0].shape[0]
    g = groups
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            pred = model(*[x[idx] for x in X])
            yb = y[idx]
            loss = mse(pred, yb) + alpha_rank * _within_state_rank_loss(pred, yb, g[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


def _pearson(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

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

    # ---- frozen h_c + op_b5 + pool -------------------------------------------------
    print("[embed] frozen h_c + B5 scores + pool ...", flush=True)
    emb: dict[str, torch.Tensor] = {}
    op_b5_by_state: dict[str, dict[str, float]] = {}
    pool_by_state: dict[str, dict] = {}
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
        enum, contrib, enab, role = b52.build_pool(st["problem"], st["schedule"], op_b5)
        # pool = list of (edit, sig, node_index, single_feat)
        pool = []
        for e in list(contrib) + list(enab):
            sig = b52._sig(e)
            ni = st["node_index"][f"operation:{e.operation_id}"]
            feat = torch.tensor(single_feature_vec(e, op_b5, role[sig]), dtype=torch.float32)
            pool.append({"e": e, "sig": sig, "ni": ni, "feat": feat,
                         "role": role[sig], "h": h_c[ni]})
        pool_by_state[iid] = {"enum": enum, "pool": pool, "contrib": len(contrib),
                              "enab": len(enab), "op_b5": op_b5}

    # ---- Phase 2: single labels (execute every pool edit) --------------------------
    single_path = OUT_DIR / "labels_single.json"
    if single_path.exists() and not args.regen:
        single_U = json.loads(single_path.read_text())
        print(f"[labels] loaded {len(single_U)} single labels from cache", flush=True)
    else:
        print("[labels] executing single edits (Frozen-Local) ...", flush=True)
        executor = AtomicCounterfactualExecutor()
        single_U: dict[str, float] = {}
        t0 = time.time()
        for inst in order:
            iid = inst["instance_id"]
            st = states[iid]
            problem, schedule = st["problem"], st["schedule"]
            base_ms = int(schedule.makespan)
            base_hash = schedule_hash(schedule)
            pdata = pool_by_state[iid]
            for rec in pdata["pool"]:
                r = _exec(executor, problem, schedule, [rec["e"]], base_ms, base_hash)
                single_U[f"{iid}::{rec['sig']}"] = r["improvement"] if r["improvement"] is not None else None
        single_path.write_text(json.dumps(single_U))
        print(f"[labels] single done: {len(single_U)} edits in {time.time()-t0:.1f}s", flush=True)

    # ---- Phase 3: pair labels (stratified sample) -----------------------------------
    pair_path = OUT_DIR / "labels_pair.jsonl"
    if pair_path.exists() and not args.regen:
        pair_rows = [json.loads(l) for l in pair_path.open()]
        print(f"[labels] loaded {len(pair_rows)} pair labels from cache", flush=True)
    else:
        print("[labels] enumerating + sampling + executing pairs ...", flush=True)
        executor = AtomicCounterfactualExecutor()
        pair_rows = []
        t0 = time.time()
        for inst in order:
            iid = inst["instance_id"]
            st = states[iid]
            problem, schedule = st["problem"], st["schedule"]
            base_ms = int(schedule.makespan)
            base_hash = schedule_hash(schedule)
            pdata = pool_by_state[iid]
            pool = pdata["pool"]
            gv = ScheduleGraphView.from_problem_schedule(problem, schedule)
            # enumerate composite-legal pairs (with type + true_add)
            cand = []
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    u, v = pool[i], pool[j]
                    if u["e"].operation_id == v["e"].operation_id:
                        continue
                    leg = check_composite_structural_legality(gv, (u["e"], v["e"]))
                    if not leg.legal:
                        continue
                    t = b52._classify_pair(u["e"], v["e"])
                    true_add = (single_U.get(f"{iid}::{u['sig']}") or 0.0) + (single_U.get(f"{iid}::{v['sig']}") or 0.0)
                    cand.append((i, j, t, true_add))
            # stratify by type, then top/mid/bottom by true_add
            by_type = collections.defaultdict(list)
            for c in cand:
                by_type[c[2]].append(c)
            sampled = []
            for t, cs in by_type.items():
                budget = PAIR_SAMPLE_BUDGET.get(t, 100)
                if len(cs) <= budget:
                    sampled.extend(cs)
                    continue
                cs_sorted = sorted(cs, key=lambda c: -c[3])
                n = len(cs_sorted)
                top = cs_sorted[:n // 4]; mid = cs_sorted[n // 4:3 * n // 4]; bot = cs_sorted[3 * n // 4:]
                n_top = budget // 3; n_bot = budget // 3
                n_mid = budget - n_top - n_bot
                s_top = random.sample(top, min(len(top), n_top))
                s_mid = random.sample(mid, min(len(mid), n_mid))
                s_bot = random.sample(bot, min(len(bot), n_bot))
                sampled.extend(s_top + s_mid + s_bot)
            for (i, j, t, _) in sampled:
                u, v = pool[i], pool[j]
                r = _exec(executor, problem, schedule, [u["e"], v["e"]], base_ms, base_hash)
                if r["improvement"] is None:
                    continue
                Uuv = r["improvement"]
                Uu = single_U.get(f"{iid}::{u['sig']}")
                Uv = single_U.get(f"{iid}::{v['sig']}")
                if Uu is None or Uv is None:
                    continue
                I_ms = Uuv - Uu - Uv
                pair_rows.append({
                    "state_id": iid, "split": inst["split"],
                    "usig": u["sig"], "vsig": v["sig"],
                    "U_uv": Uuv, "U_u": Uu, "U_v": Uv, "I_ms": I_ms,
                    "type": t, "rr": "CE" if (u["role"] != v["role"]) else ("CC" if u["role"] == "CONTRIBUTOR" else "EE"),
                })
        with pair_path.open("w") as fh:
            for r in pair_rows:
                fh.write(json.dumps(r) + "\n")
        print(f"[labels] pair done: {len(pair_rows)} pairs in {time.time()-t0:.1f}s", flush=True)

    # ---- Phase 4: build training tensors ---------------------------------------------
    print("[data] building training tensors ...", flush=True)

    # single training data (only TRAIN states for model fitting)
    def _row_to_single(iid, sig):
        pdata = pool_by_state[iid]
        for rec in pdata["pool"]:
            if rec["sig"] == sig:
                return rec
        return None

    # map state+usig -> h, feat, U_hat (single), U_true
    train_single_X, train_single_y, train_single_g = [], [], []
    val_single_X, val_single_y, val_single_g = [], [], []
    group_id = 0
    state_gid = {}
    for inst in order:
        iid = inst["instance_id"]
        state_gid[iid] = group_id
        group_id += 1
        pdata = pool_by_state[iid]
        for rec in pdata["pool"]:
            U = single_U.get(f"{iid}::{rec['sig']}")
            if U is None:
                continue
            X = torch.cat([rec["h"], rec["feat"]])
            if inst["split"] == "train":
                train_single_X.append(X); train_single_y.append(U); train_single_g.append(state_gid[iid])
            else:
                val_single_X.append(X); val_single_y.append(U); val_single_g.append(state_gid[iid])

    train_single_X = torch.stack(train_single_X); train_single_y = torch.tensor(train_single_y, dtype=torch.float32)
    train_single_g = torch.tensor(train_single_g)
    val_single_X = torch.stack(val_single_X); val_single_y = torch.tensor(val_single_y, dtype=torch.float32)
    val_single_g = torch.tensor(val_single_g)
    print(f"[data] single labels: TRAIN={train_single_X.shape[0]} VAL={val_single_X.shape[0]}", flush=True)

    # ---- train single head ------------------------------------------------------------
    print("[train] M3SingleUtilityHead ...", flush=True)
    single_head = M3SingleUtilityHead()
    single_head = _train_head(single_head, [train_single_X], train_single_y, train_single_g,
                              epochs=args.epochs, lr=args.lr, batch=args.batch, alpha_rank=0.1)
    with torch.no_grad():
        pred_tr = single_head(train_single_X)
        pred_va = single_head(val_single_X)
    single_metrics = {
        "train_pearson": _pearson(pred_tr.numpy(), train_single_y.numpy()),
        "val_pearson": _pearson(pred_va.numpy(), val_single_y.numpy()),
        "val_spearman": _pearson(np.argsort(np.argsort(pred_va.numpy())), np.argsort(np.argsort(val_single_y.numpy()))),
    }
    print(f"[M3 single] {single_metrics}", flush=True)

    # compute U_hat for every pool edit (frozen single head)
    uhat = {}  # (iid, sig) -> float
    with torch.no_grad():
        for iid, pdata in pool_by_state.items():
            for rec in pdata["pool"]:
                X = torch.cat([rec["h"], rec["feat"]]).unsqueeze(0)
                uhat[(iid, rec["sig"])] = single_head(X).item()

    # ---- pair training data (residual + direct) ----------------------------------------
    def _pair_X(iid, usig, vsig):
        pdata = pool_by_state[iid]
        rec_u = rec_v = None
        for rec in pdata["pool"]:
            if rec["sig"] == usig:
                rec_u = rec
            if rec["sig"] == vsig:
                rec_v = rec
        if rec_u is None or rec_v is None:
            return None
        st = states[iid]
        enum = pdata["enum"]
        pu = b52.parse_route_atom(usig); pv = b52.parse_route_atom(vsig)
        pstruct = b52.pair_feature_vec(st["problem"], st["schedule"], enum,
                                       rec_u["e"].operation_id, rec_u["e"].source_machine, rec_u["e"].target_machine,
                                       rec_v["e"].operation_id, rec_v["e"].source_machine, rec_v["e"].target_machine)
        uhat_u = uhat[(iid, usig)]; uhat_v = uhat[(iid, vsig)]
        return {
            "h_u": rec_u["h"], "h_v": rec_v["h"],
            "fu": rec_u["feat"], "fv": rec_v["feat"],
            "pstruct": torch.tensor(pstruct, dtype=torch.float32),
            "uhat_u": torch.tensor([uhat_u], dtype=torch.float32),
            "uhat_v": torch.tensor([uhat_v], dtype=torch.float32),
        }

    def _build_pair_data(rows, split):
        Xr, Xd, yr, yd, g = [], [], [], [], []
        for r in rows:
            if r["split"] != split:
                continue
            d = _pair_X(r["state_id"], r["usig"], r["vsig"])
            if d is None:
                continue
            Xr.append(torch.cat([d["h_u"], d["h_v"], d["fu"], d["fv"], d["pstruct"], d["uhat_u"], d["uhat_v"]]))
            Xd.append(torch.cat([d["h_u"], d["h_v"], d["fu"], d["fv"], d["pstruct"]]))
            yr.append(r["I_ms"]); yd.append(r["U_uv"])
            g.append(state_gid[r["state_id"]])
        return (torch.stack(Xr), torch.stack(Xd), torch.tensor(yr, dtype=torch.float32),
                torch.tensor(yd, dtype=torch.float32), torch.tensor(g))

    trXr, trXd, tryr, tryd, trg = _build_pair_data(pair_rows, "train")
    vaXr, vaXd, vayr, vayd, vag = _build_pair_data(pair_rows, "val")
    print(f"[data] pair labels: TRAIN={trXr.shape[0]} VAL={vaXr.shape[0]}", flush=True)

    # ---- train residual head (target I_ms) --------------------------------------------
    print("[train] M3PairResidualHead (I_ms) ...", flush=True)
    residual_head = M3PairResidualHead()
    residual_head = _train_head(residual_head, [trXr], tryr, trg, epochs=args.epochs, lr=args.lr, batch=args.batch, alpha_rank=0.1)
    with torch.no_grad():
        pr_tr = residual_head(trXr)
        pr_va = residual_head(vaXr)
    # final M3 pair prediction on VAL = U_hat(u) + U_hat(v) + I_hat
    val_upair_pred = (pr_va + vaXr[:, -2] + vaXr[:, -1]).numpy()
    residual_metrics = {
        "train_Ims_pearson": _pearson(pr_tr.numpy(), tryr.numpy()),
        "val_Ims_pearson": _pearson(pr_va.numpy(), vayr.numpy()),
        "val_Upair_pearson": _pearson(val_upair_pred, vayd.numpy()),
    }
    print(f"[M3 residual] {residual_metrics}", flush=True)

    # ---- train direct pair head (baseline, target U_pair) -----------------------------
    print("[train] DirectPairUtilityHead (baseline) ...", flush=True)
    direct_head = DirectPairUtilityHead()
    direct_head = _train_head(direct_head, [trXd], tryd, trg, epochs=args.epochs, lr=args.lr, batch=args.batch, alpha_rank=0.1)
    with torch.no_grad():
        pd_tr = direct_head(trXd)
        pd_va = direct_head(vaXd)
    direct_metrics = {
        "train_pearson": _pearson(pd_tr.numpy(), tryd.numpy()),
        "val_pearson": _pearson(pd_va.numpy(), vayd.numpy()),
    }
    print(f"[M3 direct] {direct_metrics}", flush=True)

    # ---- B5.2 appearance-I_uv control head (re-train, method B) -----------------------
    print("[train] B5.2 appearance-I_uv control head ...", flush=True)
    b52_head = b52.PairInteractionHead()
    b52X, b52y = [], []
    for split in ("train", "val"):
        for l in (TEACHER_DIR / f"pair_{split}.jsonl").open():
            r = json.loads(l)
            if r.get("I_uv") is None or r["status"] != "VALID_ANCHORED":
                continue
            if r.get("edit_kind_u") != "routing" or r.get("edit_kind_v") != "routing":
                continue
            pu = b52.parse_route_atom(r["atom_u"]); pv = b52.parse_route_atom(r["atom_v"])
            if pu is None or pv is None:
                continue
            iid = r["instance_id"]
            if iid not in states:
                continue
            st = states[iid]
            ni_u = st["node_index"].get(f"operation:{pu[0]}")
            ni_v = st["node_index"].get(f"operation:{pv[0]}")
            if ni_u is None or ni_v is None:
                continue
            enum = LegalEditEnumerator(st["problem"], st["schedule"])
            feat = b52.pair_feature_vec(st["problem"], st["schedule"], enum, pu[0], pu[1], pu[2], pv[0], pv[1], pv[2])
            hu = emb[iid][ni_u]; hv = emb[iid][ni_v]
            b52X.append(torch.cat([hu, hv, torch.tensor(feat, dtype=torch.float32)]))
            b52y.append(float(r["I_uv"]))
    b52X = torch.stack(b52X); b52y = torch.tensor(b52y, dtype=torch.float32)
    opt = torch.optim.AdamW(b52_head.parameters(), lr=args.lr, weight_decay=1e-4)
    mse = nn.MSELoss()
    for _ in range(args.epochs):
        b52_head.train(); opt.zero_grad()
        idx = torch.randperm(b52X.shape[0])[:args.batch]
        pred = b52_head(b52X[idx, :HIDDEN_DIM], b52X[idx, HIDDEN_DIM:2*HIDDEN_DIM], b52X[idx, 2*HIDDEN_DIM:])
        loss = mse(pred, b52y[idx]); loss.backward()
        torch.nn.utils.clip_grad_norm_(b52_head.parameters(), 1.0); opt.step()
    b52_head.eval()
    print(f"[B5.2 control] trained on {b52X.shape[0]} pairs", flush=True)

    # ---- save heads -------------------------------------------------------------------
    torch.save(single_head.state_dict(), OUT_DIR / "head_single.pt")
    torch.save(residual_head.state_dict(), OUT_DIR / "head_residual.pt")
    torch.save(direct_head.state_dict(), OUT_DIR / "head_direct.pt")
    torch.save(b52_head.state_dict(), OUT_DIR / "head_b52.pt")

    # ---- Phase 5: ranking evaluation -----------------------------------------------
    print("[eval] ranking all pairs + budget ...", flush=True)
    oracle = json.loads(ORACLE_DIR.joinpath("result.json").read_text())
    oracle_state = {s["state_id"]: s for s in oracle["per_state"]}

    def oracle_best_of(s):
        rows = [r for r in s["joint_rows"] if r["improvement"] is not None]
        return max(rows, key=lambda r: r["improvement"]) if rows else None

    executor = AtomicCounterfactualExecutor()
    states_rows = []

    for inst in order:
        iid = inst["instance_id"]
        st = states[iid]
        problem, schedule = st["problem"], st["schedule"]
        base_ms = int(schedule.makespan)
        base_hash = schedule_hash(schedule)
        pdata = pool_by_state[iid]
        pool = pdata["pool"]
        op_b5 = pdata["op_b5"]
        h_c = emb[iid]
        gv = ScheduleGraphView.from_problem_schedule(problem, schedule)

        # score every composite-legal pair with all methods
        scored = []  # dict: usig, vsig, iu, iv, add, b52s, m3s, directs
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                u, v = pool[i], pool[j]
                if u["e"].operation_id == v["e"].operation_id:
                    continue
                leg = check_composite_structural_legality(gv, (u["e"], v["e"]))
                if not leg.legal:
                    continue
                add = b52._score_atomic(u["e"], op_b5) + b52._score_atomic(v["e"], op_b5)
                pstruct = b52.pair_feature_vec(problem, schedule, pdata["enum"],
                                               u["e"].operation_id, u["e"].source_machine, u["e"].target_machine,
                                               v["e"].operation_id, v["e"].source_machine, v["e"].target_machine)
                pstruct_t = torch.tensor(pstruct, dtype=torch.float32)
                uu = uhat[(iid, u["sig"])]; uv = uhat[(iid, v["sig"])]
                Xr = torch.cat([h_c[u["ni"]], h_c[v["ni"]], u["feat"], v["feat"], pstruct_t,
                                torch.tensor([uu]), torch.tensor([uv])]).unsqueeze(0)
                Xd = torch.cat([h_c[u["ni"]], h_c[v["ni"]], u["feat"], v["feat"], pstruct_t]).unsqueeze(0)
                with torch.no_grad():
                    ib = b52_head(h_c[u["ni"]].unsqueeze(0), h_c[v["ni"]].unsqueeze(0), pstruct_t.unsqueeze(0)).item()
                    ir = residual_head(Xr).item()
                    idir = direct_head(Xd).item()
                scored.append({
                    "usig": u["sig"], "vsig": v["sig"], "iu": i, "iv": j,
                    "add": add, "b52s": add + LAMBDA * ib,
                    "m3s": uu + uv + LAMBDA * ir,
                    "directs": idir,
                })

        # oracle-best rank per method
        ob = oracle_best_of(oracle_state.get(iid, {}))
        ob_ranks = {"add": None, "b52": None, "m3": None, "direct": None}
        ob_in_pool = None
        if ob is not None:
            sigset = {p["sig"] for p in pool}
            if ob["u"] in sigset and ob["v"] in sigset:
                ob_in_pool = True
                # find the scored entry for oracle-best pair
                ob_entry = next((s for s in scored if (s["usig"] == ob["u"] and s["vsig"] == ob["v"]) or (s["usig"] == ob["v"] and s["vsig"] == ob["u"])), None)
                if ob_entry is not None:
                    for m in ("add", "b52", "m3", "direct"):
                        ob_ranks[m] = 1 + sum(1 for s in scored if s[SCORE_KEY[m]] > ob_entry[SCORE_KEY[m]] + 1e-12)
            else:
                ob_in_pool = False

        # budget eval (top-20 nested prefixes per method)
        budget = {}
        for m in ("add", "b52", "m3", "direct"):
            top = sorted(scored, key=lambda s: -s[SCORE_KEY[m]])[:max(PAIR_BUDGETS)]
            evs = []
            for s in top:
                r = _exec(executor, problem, schedule,
                          [pool[s["iu"]]["e"], pool[s["iv"]]["e"]], base_ms, base_hash)
                evs.append(r["improvement"])
            out = {}
            for N in PAIR_BUDGETS:
                imp = [x for x in evs[:N] if x is not None]
                best_single = max((single_U.get(f"{iid}::{p['sig']}") for p in pool if single_U.get(f"{iid}::{p['sig']}") is not None), default=None)
                out[N] = {"best": max(imp) if imp else None,
                          "n_improving": sum(1 for x in imp if x > 0),
                          "beats_single": (max(imp) if imp else -1) > (best_single or -1) if imp else False}
            budget[m] = out

        # ---- normal-M5 audit + STOP behavior -------------------------------------
        apm = set()
        for blk in st["appearance"]["blocks"]:
            if blk["block"]["block_id"] in important[iid]:
                apm.update(blk["block"]["machines"])
        contrib_tgt = {rec["e"].target_machine for rec in pool if rec["role"] == "CONTRIBUTOR"}
        normal_mid = contrib_tgt - apm
        normal_mid_enabler = [rec for rec in pool if rec["role"] == "ENABLER"
                              and rec["e"].source_machine in normal_mid]

        def _is_normal_mid_ce(s):
            u = pool[s["iu"]]; v = pool[s["iv"]]
            if u["role"] == v["role"]:
                return False
            if u["role"] == "ENABLER" and u["e"].source_machine in normal_mid:
                return True
            if v["role"] == "ENABLER" and v["e"].source_machine in normal_mid:
                return True
            return False

        nm_ce = [s for s in scored if _is_normal_mid_ce(s)]
        nm_ce_sorted = sorted(nm_ce, key=lambda s: -s["m3s"])
        nm_ce_top10 = [s for s in nm_ce_sorted
                       if 1 + sum(1 for t in scored if t["m3s"] > s["m3s"] + 1e-12) <= 10]
        nm_ce_eval = []
        for s in nm_ce_sorted[:20]:
            u = pool[s["iu"]]; v = pool[s["iv"]]
            r = _exec(executor, problem, schedule, [u["e"], v["e"]], base_ms, base_hash)
            if r["improvement"] is None:
                continue
            su = single_U.get(f"{iid}::{u['sig']}"); sv = single_U.get(f"{iid}::{v['sig']}")
            beats = r["improvement"] > max(su if su is not None else -1e18,
                                           sv if sv is not None else -1e18)
            nm_ce_eval.append({"usig": u["sig"], "vsig": v["sig"],
                               "U": r["improvement"], "beats_single": bool(beats)})

        best_single_uhat = max((uhat[(iid, rec["sig"])] for rec in pool), default=0.0)
        best_m3_pair = max((s["m3s"] for s in scored), default=0.0)
        best_pred = max(best_single_uhat, best_m3_pair)
        oracle_pos = ob is not None and ob["improvement"] > 0

        states_rows.append({
            "state_id": iid, "split": inst["split"], "base_makespan": base_ms,
            "n_contrib": pdata["contrib"], "n_enab": pdata["enab"],
            "n_pairs": len(scored), "best_single": max((single_U.get(f"{iid}::{p['sig']}") for p in pool if single_U.get(f"{iid}::{p['sig']}") is not None), default=None),
            "oracle_best": ob, "oracle_best_in_pool": ob_in_pool, "oracle_best_ranks": ob_ranks,
            "budget": budget,
            "normal_mid_machines": len(normal_mid),
            "normal_mid_enabler_edits": len(normal_mid_enabler),
            "normal_mid_ce_pairs": len(nm_ce),
            "normal_mid_ce_top10": len(nm_ce_top10),
            "normal_mid_ce_positive": sum(1 for e in nm_ce_eval if e["U"] > 0),
            "normal_mid_ce_beats_single": sum(1 for e in nm_ce_eval if e["beats_single"]),
            "best_single_uhat": best_single_uhat, "best_m3_pair": best_m3_pair,
            "best_pred_utility": best_pred, "oracle_positive": oracle_pos,
            "pred_act": best_pred > 0,
        })
        print(f"[eval] {iid:26s} n_pairs={len(scored)} ob_in_pool={ob_in_pool} "
              f"rank(add={ob_ranks['add']},b52={ob_ranks['b52']},m3={ob_ranks['m3']},direct={ob_ranks['direct']}) "
              f"nm_mid={len(normal_mid)} nm_ce_top10={len(nm_ce_top10)} pred_util={best_pred:.1f}",
              flush=True)

    # ---- aggregate ------------------------------------------------------------------
    def _med(xs):
        return float(np.median(xs)) if xs else None

    agg = {"n_states": len(states_rows)}
    agg["single_metrics"] = single_metrics
    agg["residual_metrics"] = residual_metrics
    agg["direct_metrics"] = direct_metrics
    agg["n_single_labels"] = len(single_U)
    agg["n_pair_labels"] = len(pair_rows)
    # label distribution
    U = np.array([r["U_uv"] for r in pair_rows]); Ims = np.array([r["I_ms"] for r in pair_rows])
    agg["label_dist"] = {
        "pair_U_pos": int((U > 0).sum()), "pair_U_zero": int((U == 0).sum()), "pair_U_neg": int((U < 0).sum()),
        "Ims_pos": int((Ims > 1e-9).sum()), "Ims_zero": int((abs(Ims) <= 1e-9).sum()), "Ims_neg": int((Ims < -1e-9).sum()),
        "pair_CC": sum(1 for r in pair_rows if r["rr"] == "CC"),
        "pair_CE": sum(1 for r in pair_rows if r["rr"] == "CE"),
        "pair_EE": sum(1 for r in pair_rows if r["rr"] == "EE"),
    }
    agg["oracle_best_in_pool"] = sum(1 for r in states_rows if r["oracle_best_in_pool"])
    for m in ("add", "b52", "m3", "direct"):
        ranks = [r["oracle_best_ranks"][m] for r in states_rows if r["oracle_best_ranks"][m] is not None]
        agg[f"oracle_best_rank_median_{m}"] = _med(ranks)
        for Kk in (1, 3, 5, 10):
            rankable = [r for r in states_rows if r["oracle_best_ranks"][m] is not None]
            agg[f"P@{Kk}_{m}"] = sum(1 for r in rankable if r["oracle_best_ranks"][m] <= Kk) / len(rankable) if rankable else None
    # strong-interaction median rank
    for m in ("add", "b52", "m3", "direct"):
        strong = [r["oracle_best_ranks"][m] for r in states_rows
                  if r["oracle_best"] is not None and r["oracle_best_ranks"][m] is not None
                  and abs(r["oracle_best"]["I_ms"]) >= 10]
        agg[f"strong_interaction_median_rank_{m}"] = _med(strong)
    # budget aggregates
    agg["budgets"] = {}
    for N in PAIR_BUDGETS:
        agg["budgets"][N] = {}
        for m in ("add", "b52", "m3", "direct"):
            agg["budgets"][N][m] = {
                "positive_states": sum(1 for r in states_rows if r["budget"][m][N]["n_improving"] > 0),
                "beats_single_states": sum(1 for r in states_rows if r["budget"][m][N]["beats_single"]),
                "best_joint": max((r["budget"][m][N]["best"] for r in states_rows if r["budget"][m][N]["best"] is not None), default=None),
            }
    # normal-M5 capability aggregate
    agg["normal_m5"] = {
        "normal_mid_machines_total": sum(r["normal_mid_machines"] for r in states_rows),
        "normal_mid_enabler_edits_total": sum(r["normal_mid_enabler_edits"] for r in states_rows),
        "normal_mid_ce_pairs_total": sum(r["normal_mid_ce_pairs"] for r in states_rows),
        "normal_mid_ce_top10_total": sum(r["normal_mid_ce_top10"] for r in states_rows),
        "normal_mid_ce_positive_total": sum(r["normal_mid_ce_positive"] for r in states_rows),
        "normal_mid_ce_beats_single_total": sum(r["normal_mid_ce_beats_single"] for r in states_rows),
    }
    # STOP behavior aggregate (U(STOP)=0)
    n_oracle_pos = sum(1 for r in states_rows if r["oracle_positive"])
    n_pred_act = sum(1 for r in states_rows if r["pred_act"])
    correct_act = sum(1 for r in states_rows if r["pred_act"] and r["oracle_positive"])
    correct_stop = sum(1 for r in states_rows if not r["pred_act"] and not r["oracle_positive"])
    agg["stop"] = {
        "n_oracle_positive": n_oracle_pos,
        "n_pred_act": n_pred_act,
        "correct_act": correct_act,
        "correct_stop": correct_stop,
        "decision_accuracy": (correct_act + correct_stop) / len(states_rows) if states_rows else None,
    }

    (OUT_DIR / "result.json").write_text(json.dumps(agg, indent=2, default=str))
    for r in states_rows:
        (OUT_DIR / f"state_{r['state_id']}.json").write_text(json.dumps(r, indent=2, default=str))

    print("\n==================== M3 MAKESPAN UTILITY ====================")
    print(f"  single: {agg['single_metrics']}")
    print(f"  residual: {agg['residual_metrics']}")
    print(f"  direct: {agg['direct_metrics']}")
    print(f"  label_dist: {agg['label_dist']}")
    print(f"  oracle-best in pool: {agg['oracle_best_in_pool']}/{agg['n_states']}")
    for m in ("add", "b52", "m3", "direct"):
        print(f"  [{m}] oracle-best rank median={agg[f'oracle_best_rank_median_{m}']} "
              f"strong-I median={agg[f'strong_interaction_median_rank_{m}']} "
              f"P@1={agg[f'P@1_{m}']} P@3={agg[f'P@3_{m}']} P@5={agg[f'P@5_{m}']} P@10={agg[f'P@10_{m}']}")
    for N in PAIR_BUDGETS:
        row = agg["budgets"][N]
        print(f"  budget={N:2d}: " + "  ".join(f"{m}(pos={row[m]['positive_states']},beat={row[m]['beats_single_states']},best={row[m]['best_joint']})" for m in ("add", "b52", "m3", "direct")))
    print(f"  normal_m5: {agg['normal_m5']}")
    print(f"  stop: {agg['stop']}")
    print(f"  Saved: {OUT_DIR / 'result.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--regen", action="store_true", help="regenerate cached labels")
    args = ap.parse_args()
    main(args)
