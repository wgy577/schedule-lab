"""Canonical M3 ACT/STOP gate (proposal-pooled statistics).

VERBATIM extraction (no semantic change):
  - _old_base_arr / pooled_gate_input / dual_final_order / DualGate /
    train_gate_v2 / gate_report                <- r3
  - train_gate_r4 (act_primary re-derivation on corrected-memory scorer)  <- R4
Frozen decision: act_primary weights ACT:STOP = 2:1 (false-stop is primary --
a balanced CE that upweights the minority STOP was proven wrong in R3).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import (
    EPOCHS,
    GATE_FEAT_DIM,
    GATE_HIDDEN,
    K_C,
    K_R,
    LR,
)


def _old_base_arr(prop_feats):
    """Frozen utility base per proposal (single->idx296, pair->idx298)."""
    if isinstance(prop_feats, np.ndarray):
        return np.where(prop_feats[:, 299] < 0.5, prop_feats[:, 296], prop_feats[:, 298])
    return torch.where(prop_feats[:, 299] < 0.5, prop_feats[:, 296], prop_feats[:, 298])


def pooled_gate_input(state_feat, logit_pos, rank, old_base, K_r=K_R, K_c=K_C):
    """State-feat + proposal pooled statistics (directive Section 6).

    Returns (x[GATE_FEAT_DIM], R, C, union) where R/C are python index lists.
    """
    N = logit_pos.shape[0]
    with torch.no_grad():
        k3 = min(3, N)
        top3_r = torch.topk(rank, k3).values
        top3_l = torch.topk(logit_pos, k3).values
        max_r = rank.max()
        mean3_r = top3_r.mean()
        max_l = logit_pos.max()
        mean3_l = top3_l.mean()
        n_cls_pos = float((logit_pos > 0).sum())
        R = torch.topk(rank, min(K_r, N)).indices.tolist()
        C = torch.topk(logit_pos, min(K_c, N)).indices.tolist()
        union = sorted(set(R) | set(C))
        agr = 1.0 if int(torch.argmax(rank)) in set(C) else 0.0
        old_max = float(old_base.max()) if hasattr(old_base, "max") else float(max(old_base))
    x = torch.cat([
        torch.as_tensor(state_feat, dtype=torch.float32),
        torch.tensor([float(max_r), float(mean3_r), float(max_l), float(mean3_l),
                      n_cls_pos, float(len(union)), agr, old_max], dtype=torch.float32),
    ])
    return x, R, C, union


def dual_final_order(union, rank, logit_pos):
    """rank_head order within shortlist; usefulness only as tie-break."""
    return sorted(union, key=lambda k: (float(rank[k]), float(logit_pos[k])), reverse=True)


# ---------------------------------------------------------------------------
# ACT/STOP gate (proposal-pooled input)  [r3]
# ---------------------------------------------------------------------------
class DualGate(nn.Module):
    def __init__(self, in_dim=GATE_FEAT_DIM, hidden=GATE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x).squeeze(0)           # [2]; 0=ACT, 1=STOP


def train_gate_v2(scorer, state_examples, use_mem, seed=0, epochs=EPOCHS, lr=LR,
                  K_r=K_R, K_c=K_C, mode="act_primary"):
    """mode: 'act_primary' (false-stop primary, Section 8: err toward ACT) or
    'balanced' (minority-STOP upweight; keeps ACC lean but raises false-stop)."""
    Xs, ys = [], []
    for ex in state_examples:
        with torch.no_grad():
            mem = ex["mem_feats"] if use_mem else None
            logit_pos, rank, _ = scorer(ex["prop_feats"], mem, ex["state_feat"])
        oldbase = torch.as_tensor(ex["old_base"], dtype=torch.float32)
        x, *_ = pooled_gate_input(ex["state_feat"], logit_pos, rank, oldbase, K_r, K_c)
        y = 1.0 if (ex["true_U"] > 0).any() else 0.0
        Xs.append(x)
        ys.append(y)
    X = torch.stack(Xs)
    y = torch.tensor(ys)
    n_act = int(y.sum())
    n_stop = len(y) - n_act
    tgt = 1 - y  # class 0=ACT, 1=STOP
    if mode == "act_primary":
        # ACT is the majority (28/40) and false-stop is PRIMARY (Section 8):
        # upweight ACT-class samples so the gate refuses to miss a true-ACT.
        # A "balanced" weight here would upweight the minority STOP and push the
        # model toward STOP -- the exact failure observed in the first R3 full run.
        w = torch.tensor([2.0, 1.0])
    else:
        # balanced CE: weight ACT class by n_stop/n_act, STOP by n_act/n_stop
        w = torch.tensor([float(n_stop / max(n_act, 1)), float(n_act / max(n_stop, 1))])
    torch.manual_seed(seed)
    gate = DualGate()
    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        logits = gate(X)
        loss = torch.nn.functional.cross_entropy(logits, tgt.long(), weight=w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
        opt.step()
    gate.eval()
    return gate, X, y, {"n_act": n_act, "n_stop": n_stop, "mode": mode, "weights": [float(w[0]), float(w[1])]}


def gate_report(gate, X, y):
    """accuracy / ACT recall / STOP recall / false-stop(primary) / false-act."""
    with torch.no_grad():
        pred_cls = gate(X).argmax(1).numpy()          # 0=ACT, 1=STOP
    y = y.numpy()
    pred_act = pred_cls == 0
    true_act = y == 1
    acc = float((pred_act == true_act).mean())
    act_recall = float((pred_act & true_act).sum() / max(true_act.sum(), 1))
    stop_recall = float((~pred_act & ~true_act).sum() / max((~true_act).sum(), 1))
    false_stop = float(((~pred_act) & true_act).sum() / max(true_act.sum(), 1))
    false_act = float((pred_act & (~true_act)).sum() / max((~true_act).sum(), 1))
    return {"gate_accuracy": acc, "act_recall": act_recall, "stop_recall": stop_recall,
            "false_stop": false_stop, "false_act": false_act}


def train_gate_r4(scorer_mem, state_examples, mode="act_primary", seed=0):
    """R4 canonical gate: re-derived on the corrected-memory scorer (mem present)."""
    Xs, ys = [], []
    for ex in state_examples:
        with torch.no_grad():
            logit_pos, rank, _ = scorer_mem(ex["prop_feats"], ex["prog_mem_feats"], ex["state_feat"])
        oldbase = torch.as_tensor(ex["old_base"], dtype=torch.float32)
        x, *_ = pooled_gate_input(ex["state_feat"], logit_pos, rank, oldbase, K_R, K_C)
        y = 1.0 if (ex["true_U"] > 0).any() else 0.0
        Xs.append(x)
        ys.append(y)
    X = torch.stack(Xs)
    y = torch.tensor(ys)
    n_act = int(y.sum())
    n_stop = len(y) - n_act
    tgt = 1 - y
    w = torch.tensor([2.0, 1.0]) if mode == "act_primary" else \
        torch.tensor([float(n_stop / max(n_act, 1)), float(n_act / max(n_stop, 1))])
    torch.manual_seed(seed)
    gate = DualGate()
    opt = torch.optim.AdamW(gate.parameters(), lr=LR, weight_decay=1e-4)
    for _ in range(EPOCHS):
        opt.zero_grad()
        logits = gate(X)
        loss = torch.nn.functional.cross_entropy(logits, tgt.long(), weight=w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
        opt.step()
    gate.eval()
    return gate, X, y, {"n_act": n_act, "n_stop": n_stop, "mode": mode,
                        "weights": [float(w[0]), float(w[1])],
                        "gate_report": gate_report(gate, X, y)}