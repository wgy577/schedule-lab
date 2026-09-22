"""T2-B-GPU -- packed GPU engine + preflight + success gate.

Independent of the canonical Mac/CPU path.  `packed_gpu_update_joint` is the
dedicated-GPU update: it replicates `joint_grpo.grpo_update_joint` (stagewise
A2/A3 credit, T2-B/R14 VERBATIM) on PACKED CUDA tensors -- one M3 forward over
the concatenated [sum M_b, D] proposal features, one M2 forward over the
concatenated root features, and a padded [B, Pmax] matrix so the softmax always
covers each state's FULL valid action set (proposals + STOP; -inf padding never
leaks mass).

Mathematical parity is retained exactly: per-trajectory-equal weighting, same-
state group-relative A3/A2, per-step KL means (kl_ref_m3 / kl_old / kl_m2), a
single AdamW (one optimizer per update call, lr = stage lr, weight_decay 0),
E epochs, clip 0.20, clip_grad_norm 10.0, torch.manual_seed before each update.

It is wired into the canonical Stage-3 loop by a runtime hook:
    JG.grpo_update_joint = t2b_gpu_trainer.packed_gpu_update_joint
(restored in a finally) -- no source file in the canonical path is modified.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from typing import Any

import torch

from causal_schedule_lab.m3 import config as C
from causal_schedule_lab.m3 import joint_grpo as JG

# Runtime overrides the canonical loop cannot pass (it calls the update by
# bare name).  The runner sets these before hooking `JG.grpo_update_joint`.
_DEVICE_OVERRIDE = None
_MAX_BATCH = None
_AMP_MODE = None


def set_runtime(device=None, max_batch=None, amp=None):
    global _DEVICE_OVERRIDE, _MAX_BATCH, _AMP_MODE
    if device is not None:
        _DEVICE_OVERRIDE = device
    if max_batch is not None:
        _MAX_BATCH = int(max_batch)
    if amp is not None:
        _AMP_MODE = amp


def _gpu_policy(jpol, dev):
    """Deep GPU copy of the trainable policy (tiny).  The CPU jpol is untouched
    until the updated params are copied back at the end."""
    m3 = copy.deepcopy(jpol.m3).to(dev)
    m2 = copy.deepcopy(jpol.m2).to(dev)
    return m3, m2


def _copy_trainable_back(m3g, m2g, jpol, stage):
    with torch.no_grad():
        if stage in ("B", "C"):            # train_m3
            for pg, pc in zip(m3g.parameters(), jpol.m3.parameters()):
                if pg.requires_grad:
                    pc.copy_(pg.detach().cpu())
        if stage in ("A", "C"):            # train_m2
            for pg, pc in zip(m2g.parameters(), jpol.m2.parameters()):
                if pg.requires_grad:
                    pc.copy_(pg.detach().cpu())


def packed_gpu_update_joint(jpol, groups, stage="C", T=None, eps=None,
                            clip_eps=None, seeds=None, epochs=None,
                            log_prefix="[t2bgpu]", credit="stagewise",
                            device=None, max_batch=None, amp=None):
    """Packed GPU joint GRPO update -- signature-compatible with
    `joint_grpo.grpo_update_joint` and mathematically equivalent for the
    stagewise credit (the only credit the GPU edition uses).

    Returns the same per-update dict schema the canonical loop consumes
    (n_informative_trajectories / n_steps / epochs / n_informative_trajectories_m2
    / final_kl_m3 / final_kl_m2) plus a `_monitor` dict with packed-batch stats.
    """
    T = float(T if T is not None else C.TO1_R13_TEMP)
    eps = float(eps if eps is not None else C.TO1_R13_MIX_EPS)
    clip_eps = float(clip_eps if clip_eps is not None else C.TO1_R13_CLIP_EPS)
    epochs = int(epochs if epochs is not None else C.TO1_R13_UPDATE_EPOCHS)
    if credit != "stagewise":
        raise ValueError("T2-B-GPU only supports credit='stagewise' (frozen)")
    train_m2 = stage in ("A", "C")
    train_m3 = stage in ("B", "C")
    lr = (float(C.TO1_R13_LR_M3) if train_m3 else float(C.TO1_R13_LR_M2))
    lambda_m2 = float(C.TO1_R14_LAMBDA_M2)
    beta_m3 = float(C.TO1_R14_BETA_M3)
    beta_m2 = float(C.TO1_R14_BETA_M2)
    temp_m2 = float(C.TO1_R13_TEMP_M2)
    alpha_m2 = float(C.TO1_R13_ALPHA_M2)
    dev = torch.device(device or _DEVICE_OVERRIDE or JG.get_device())
    max_batch = int(max_batch) if max_batch is not None else _MAX_BATCH
    amp = amp or _AMP_MODE

    # ---- flatten groups -> trajectories -> steps -----------------------------
    stems = []            # light step atoms (CPU tensors kept as-is)
    n_traj = 0
    n_info3_traj = 0
    n_info2_traj = 0
    for g in groups:
        inf = bool(g.get("informative"))
        inf2 = bool(g.get("info2", inf))
        for tr in g.get("trajs", []):
            sts = tr["steps"]
            if inf and len(sts):
                n_info3_traj += 1
            if inf2 and len(sts):
                n_info2_traj += 1
            n_traj += 1
            for rec in sts:
                rec.setdefault("inf3", inf)
                rec.setdefault("inf2", inf2)
                rec.setdefault("adv3", float(rec.get("adv3", 0.0)))
                rec.setdefault("adv2", float(rec.get("adv2", 0.0)))
            for si, rec in enumerate(sts):
                stems.append(_Step(
                    traj=n_traj - 1, inf3=bool(rec["inf3"]),
                    inf2=bool(rec["inf2"]),
                    adv3=float(rec["adv3"]), adv2=float(rec["adv2"]),
                    a=int(rec["a"]), logp_old=float(rec["logp_old"]),
                    F=rec["F_pool"], evid=rec.get("evid"),
                    sf=rec["sf_t"], ps=rec["pool_stats"],
                    logits_old=rec["logits_old"], m2=rec.get("m2_rec")))
    n_steps = len(stems)
    monitor = {"stage": stage, "credit": "stagewise", "device": str(dev),
               "n_groups": len(groups), "n_trajectories": n_traj,
               "n_steps": n_steps, "n_info3_trajectories": n_info3_traj,
               "n_info2_trajectories": n_info2_traj}

    if n_steps == 0:
        return {"n_informative_trajectories": 0, "n_trajectories": n_traj,
                "n_steps": 0, "epochs": [], "reason": "no_steps",
                "n_informative_trajectories_m2": 0, "final_kl_m3": None,
                "final_kl_m2": None, "_monitor": monitor}

    m3g, m2g = _gpu_policy(jpol, dev)
    coll = _pack_steps(stems, dev)
    m2rec = _pack_m2(stems, dev) if train_m2 else None
    if m2rec is not None:
        m2rec["net"] = m2g.net

    params = []
    if train_m3:
        params += [p for p in m3g.parameters() if p.requires_grad]
    if train_m2:
        params += [p for p in m2g.parameters() if p.requires_grad]
    if not params:
        return {"n_informative_trajectories": 0, "n_trajectories": n_traj,
                "n_steps": n_steps, "epochs": [], "reason": "no_trainable_params",
                "n_informative_trajectories_m2": n_info2_traj,
                "final_kl_m3": None, "final_kl_m2": None, "_monitor": monitor}
    opt = torch.optim.AdamW([{"params": params, "lr": lr}], weight_decay=0.0)
    torch.manual_seed(0 if seeds is None else int(seeds))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0 if seeds is None else int(seeds))

    def _epoch(backward):
        B = coll["B"]; Pmax = coll["Pmax"]
        F_all, Ev_all = coll["F_all"], coll["Ev_all"]
        sf_ps = coll["sf_ps"]
        Ms, Ns, a_col, logp_old = (coll["Ms"], coll["Ns"],
                                   coll["a"], coll["logp_old"])
        adv3, adv2, inf3, inf2 = (coll["adv3"], coll["adv2"],
                                  coll["inf3"], coll["inf2"])
        traj, w3, n_s3 = coll["traj"], coll["w3"], coll["n_s3"]

        # M3 forward -- frozen r6 on the concatenated pools; trainable residual
        F_ev = torch.cat([F_all, Ev_all], dim=-1)
        with torch.no_grad():
            z_all = m3g.r6.prop_scores(F_all)                  # [sum M]
            bs_all = m3g.r6.stop_head(sf_ps).reshape(-1)       # [B]
        # T2-C additive dispatch: residual policies (C1/C2) carry
        # `residual_delta_prop` and do the mode math (large=identity, controlled=
        # cap*tanh) inside the module; the CANONICAL classes (C0 tiny +
        # M3TemporalEvidencePolicy / M3RollingGRPOPolicy) have no such method and
        # fall through to the exact canonical line below -> C0 is bit-identical
        # to T2-B-GPU.  dstop stays the canonical bounded line in ALL modes (R7
        # STOP-collapse guard -- the STOP residual is never un-capped).
        if hasattr(m3g, "residual_delta_prop"):
            dprop = m3g.residual_delta_prop(F_ev)              # [sum M]
        else:
            dprop = m3g.alpha_prop * torch.tanh(
                m3g.resid_prop(F_ev).squeeze(-1))              # [sum M]
        dstop = m3g.alpha_stop * torch.tanh(
            m3g.resid_stop(sf_ps).squeeze(-1))                 # [B]
        prop_all = z_all + dprop
        stop_all = bs_all + dstop
        # T2-C residual-scale metrics (last-epoch forward -> monitor["residual"])
        if dprop.numel():
            _d = dprop.detach()
            _z = z_all.detach()
            monitor["residual_last"] = {
                "dprop_mean_abs": float(_d.abs().mean()),
                "dprop_max_abs": float(_d.abs().max()),
                "dprop_norm": float(_d.norm(p=2)),
                "base_std": float(_z.std()) if _z.numel() else 0.0,
                "final_std": float((_z + _d).std()) if _z.numel() else 0.0,
                "alpha_prop": float(m3g.alpha_prop),
                "cap": (float(m3g.cap) if getattr(m3g, "cap", None) is not None
                        else None),
                "sft_ratio": (float((_d.abs() / (_z.abs() + 1e-6)).mean())
                              if _z.numel() else 0.0),
            }

        logitsP = torch.full((B, Pmax), float("-inf"), dtype=torch.float32,
                             device=dev)
        baseP = torch.full_like(logitsP, float("-inf"))
        atom_r = coll["atom_r"]                                # flat row id
        atom_c = coll["atom_c"]                                # flat col id
        logitsP[atom_r, atom_c] = prop_all
        baseP[atom_r, atom_c] = z_all
        logitsP[torch.arange(B, device=dev), Ms] = stop_all
        baseP[torch.arange(B, device=dev), Ms] = bs_all
        # stop column index M_s: row i has valid cols [0..M_s]
        valid = coll["valid"]
        Nv = coll["Ns"].reshape(-1, 1)
        div = torch.where(valid, Nv, torch.ones_like(Nv))

        def _pmf(zP):
            return (1.0 - eps) * torch.softmax(zP / T, dim=-1) + \
                (eps / div) * valid

        # safe-log: padded columns carry p == 0, so p*log(p) would be 0*-inf
        # in value AND 0/0 in its backward (Log at exactly 0).  Instead feed
        # pads an argument of 1 to the log: pp*valid + (1-valid) is 1 on pads
        # (p*log1 == 0, derivative 0) and pp on the valid columns (p has the
        # eps/(M+1) floor, so pp>0 and the clamp is a no-op there).  This is
        # value- and gradient-identical to the canonical per-state KL/entropy,
        # which never sees padding at all.
        vm_f = valid.to(dtype=torch.float32)

        def _lsafe(pp):
            return torch.log(pp * vm_f + (1.0 - vm_f))

        p_new = _pmf(logitsP)
        p_base = _pmf(baseP)
        p_old = _pmf(coll["oldP"])
        lp_new = torch.log(torch.gather(p_new, 1, a_col.reshape(-1, 1))
                           .squeeze(1).clamp_min(1e-12))
        ratio = torch.exp(lp_new - logp_old)

        obj3 = torch.min(ratio * adv3, torch.clamp(ratio, 1.0 - clip_eps,
                                                   1.0 + clip_eps) * adv3)
        nclip = int(((ratio > 1.0 + clip_eps) | (ratio < 1.0 - clip_eps))[inf3]
                    .sum())
        kl_ref3 = (p_new * (_lsafe(p_new) - _lsafe(p_base))).sum(dim=-1)
        kl_old = (p_new * (_lsafe(p_new) - _lsafe(p_old))).sum(dim=-1)
        ent = -(p_new * _lsafe(p_new)).sum(dim=-1)

        loss = torch.zeros((), dtype=torch.float32, device=dev)
        m2_metrics = {"n_s2_trajs": 0, "obj2": 0.0, "kl_m2": 0.0, "n_pool_steps": 0}
        if train_m3 and n_s3 > 0:
            loss = loss - (obj3 * w3).sum() / float(n_s3)
        if train_m2 and m2rec is not None:
            m2_metrics = _m2_loss(m2rec, coll, clip_eps, lambda_m2, beta_m2,
                                  dev)
            loss = loss + m2_metrics["m2_loss_term"]
        if train_m3 and B > 0:
            loss = loss + beta_m3 * kl_ref3.mean()
        # canonical beta_m3 term is added unconditionally when train_m3 & steps

        if backward:
            opt.zero_grad()
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(params, 10.0))
            opt.step()
        else:
            gn = float(max((torch.norm(p.grad) for p in params
                            if p.grad is not None), default=0.0))

        return {
            "loss": float(loss.detach()),
            "n_info_steps": int(inf3.sum()),
            "kl_ref_m3": float(kl_ref3.detach().mean()) if B else 0.0,
            "kl_old": float(kl_old.detach().mean()) if B else 0.0,
            "entropy": float(ent.detach().mean()) if B else 0.0,
            "grad_norm": gn,
            "kl_m2": m2_metrics.get("kl_m2", 0.0),
            "clip_frac_m3": float(nclip) / max(n_info3_traj, 1),
            "ratio_mean": (float(ratio[inf3].detach().mean())
                           if inf3.any() else 0.0),
            "ratio_std": (float(ratio[inf3].detach().std())
                          if inf3.any() else 0.0),
            "adv_mean": (float(adv3[inf3].detach().mean())
                         if inf3.any() else 0.0),
            "adv_std": (float(adv3[inf3].detach().std())
                        if inf3.any() else 0.0),
            "n_s2_trajs": m2_metrics.get("n_s2_trajs", 0),
        }

    per_epoch = []
    has_s3 = train_m3 and n_info3_traj > 0
    has_s2 = train_m2 and m2rec is not None and bool((coll["inf2"]
                                                      & m2rec["has_draws"]).any())
    if not has_s3 and not has_s2:
        # dry cycle: no informative s3/s2 trajectories.  Keep the same key
        # contract as a real epoch so per_epoch[-1] consumers below never
        # KeyError on a stub tail (grad_norm/ratio_mean/entropy = 0.0).
        per_epoch.append({"epoch": 0, "loss": 0.0,
                          "n_informative_trajectories": 0, "n_steps": n_steps,
                          "clip_frac_m3": 0.0, "kl_ref_m3": 0.0, "kl_m2": 0.0,
                          "grad_norm": 0.0, "ratio_mean": 0.0, "entropy": 0.0,
                          "reason": "no_informative_trajectories",
                          "stopped_stale": False})
    else:
        for ep in range(epochs):
            gm = _epoch(backward=True)
            per_epoch.append({
                "epoch": ep, "loss": gm["loss"],
                "n_informative_trajectories": n_info2_traj,
                "n_steps": n_steps, "n_info_steps": n_info2_traj,
                "grad_norm": gm["grad_norm"],
                "kl_ref_m3": gm["kl_ref_m3"], "kl_old": gm["kl_old"],
                "kl_m2": gm["kl_m2"], "clip_frac_m3": gm["clip_frac_m3"],
                "n_informative_trajectories_m2": n_info2_traj,
                "ratio_mean": gm["ratio_mean"], "ratio_std": gm["ratio_std"],
                "adv_mean": gm["adv_mean"], "adv_std": gm["adv_std"],
                "entropy": gm["entropy"],
                "stopped_stale": False, "reason": ""})

    _copy_trainable_back(m3g, m2g, jpol, stage)
    if monitor.get("residual_last") is not None:
        monitor["residual"] = monitor["residual_last"]
        monitor.pop("residual_last", None)
    # NOTE: all per_epoch[-1] reads are defensive (.get) -- a dry-cycle stub or
    # any future non-epoch tail must never KeyError the tail assembly, regardless
    # of which archive revision an instance carries (each epoch dict is required
    # to carry grad_norm/ratio_mean/entropy/kl_ref_m3/kl_m2; a stub may be thinner).
    _last = per_epoch[-1] if per_epoch else {}
    monitor.update({
        "grad_norm_last": _last.get("grad_norm", 0.0),
        "ratio_last": _last.get("ratio_mean", 0.0),
        "entropy_last": _last.get("entropy", 0.0),
        "Pmax": coll["Pmax"], "sum_M": int(coll["atom_r"].numel()),
        "max_batch": max_batch, "amp": amp,
    })
    return {"n_informative_trajectories": n_info3_traj,
            "n_trajectories": n_traj, "n_steps": n_steps, "epochs": per_epoch,
            "n_informative_trajectories_m2": n_info2_traj,
            "final_kl_m3": _last.get("kl_ref_m3", 0.0),
            "final_kl_m2": _last.get("kl_m2", 0.0),
            "_monitor": monitor}


class _Step:
    __slots__ = ("traj", "inf3", "inf2", "adv3", "adv2", "a", "logp_old",
                 "F", "evid", "sf", "ps", "logits_old", "m2")

    def __init__(self, traj, inf3, inf2, adv3, adv2, a, logp_old,
                 F, evid, sf, ps, logits_old, m2):
        self.traj = traj
        self.inf3 = inf3
        self.inf2 = inf2
        self.adv3 = adv3
        self.adv2 = adv2
        self.a = a
        self.logp_old = logp_old
        self.F = F
        self.evid = evid
        self.sf = sf
        self.ps = ps
        self.logits_old = logits_old
        self.m2 = m2


def _ten(x, dev, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype).to(dev)


def _pack_steps(stems, dev):
    B = len(stems)
    n_traj = int(max(s.traj for s in stems)) + 1
    a = torch.zeros(B, dtype=torch.long, device=dev)
    logp_old = torch.zeros(B, dtype=torch.float32, device=dev)
    adv3 = torch.zeros(B, dtype=torch.float32, device=dev)
    adv2 = torch.zeros(B, dtype=torch.float32, device=dev)
    inf3 = torch.zeros(B, dtype=torch.bool, device=dev)
    inf2 = torch.zeros(B, dtype=torch.bool, device=dev)
    traj = torch.zeros(B, dtype=torch.long, device=dev)
    Ms = torch.zeros(B, dtype=torch.long, device=dev)
    Ns = torch.zeros(B, dtype=torch.float32, device=dev)
    F_parts, Ev_parts, sf_parts, ps_parts = [], [], [], []
    old_cols = []
    off = 0
    offs = []
    for i, s in enumerate(stems):
        M = int(len(s.F)) if s.F is not None else 0
        a[i] = int(s.a)
        logp_old[i] = float(s.logp_old)
        adv3[i] = float(s.adv3)
        adv2[i] = float(s.adv2)
        inf3[i] = bool(s.inf3)
        inf2[i] = bool(s.inf2)
        traj[i] = int(s.traj)
        Ms[i] = M
        Ns[i] = float(M + 1)
        offs.append(off)
        F_parts.append(_ten(s.F, dev))
        if s.evid is not None and len(s.evid) == M:
            Ev_parts.append(_ten(s.evid, dev))
        else:
            Ev_parts.append(torch.zeros(M, int(C.TO1_R19_EVID_DIM), device=dev))
        sf_parts.append(_ten(s.sf, dev).reshape(-1))
        ps_parts.append(_ten(s.ps, dev).reshape(-1))
        old = _ten(s.logits_old, dev).reshape(-1) if s.logits_old is not None \
            else torch.zeros(M + 1, device=dev)
        old_cols.append(old)
        off += M
    F_all = torch.cat(F_parts, 0)
    Ev_all = torch.cat(Ev_parts, 0)
    sf_ps = torch.cat([torch.stack(sf_parts), torch.stack(ps_parts)], dim=-1)

    Pmax = int(Ns.max())
    oldP = torch.full((B, Pmax), float("-inf"), dtype=torch.float32, device=dev)
    for i in range(B):
        oldv = old_cols[i]
        m = int(Ms[i])
        oldP[i, :m] = oldv[:m]
        oldP[i, m] = oldv[m]

    # flat atom row/col for proposal columns
    atom_r = torch.cat([torch.full((int(m),), i, dtype=torch.long, device=dev)
                        for i, m in enumerate(Ms.tolist())])
    atom_c = torch.cat([torch.arange(int(m), dtype=torch.long, device=dev)
                        for m in Ms.tolist()])

    colm = torch.arange(Pmax, device=dev).reshape(1, Pmax).expand(B, Pmax)
    valid = colm <= Ms.reshape(-1, 1)

    cnt3 = torch.zeros(n_traj, dtype=torch.float32, device=dev)
    cnt3.scatter_add_(0, traj[inf3], torch.ones(int(inf3.sum()), device=dev))
    w3 = torch.zeros(B, dtype=torch.float32, device=dev)
    sel = torch.nonzero(inf3).view(-1)
    w3[sel] = 1.0 / cnt3[traj[sel]].clamp(min=1.0)
    n_s3 = int((cnt3 > 0).sum())

    return {"B": B, "Pmax": Pmax, "n_traj": n_traj, "F_all": F_all,
            "Ev_all": Ev_all, "sf_ps": sf_ps, "a": a, "Ms": Ms, "Ns": Ns,
            "logp_old": logp_old, "adv3": adv3, "adv2": adv2, "inf3": inf3,
            "inf2": inf2, "traj": traj, "w3": w3, "n_s3": n_s3,
            "atom_r": atom_r, "atom_c": atom_c, "valid": valid,
            "oldP": oldP}


def _pack_m2(stems, dev):
    """Collate per-step M2 candidate features + draws + pool masks."""
    cand_rows = []             # global candidate index -> (step_idx, local_idx)
    cand_parts, base_parts = [], []
    step_cand_off = -torch.ones(len(stems), dtype=torch.long, device=dev)
    step_cand_n = torch.zeros(len(stems), dtype=torch.long, device=dev)
    step_has_m2 = torch.zeros(len(stems), dtype=torch.bool, device=dev)
    pool_step = []             # per step: global cand rows of pool_mask
    draw = []
    has_draws = torch.zeros(len(stems), dtype=torch.bool, device=dev)
    c_off = 0
    for i, s in enumerate(stems):
        m2 = s.m2
        if m2 is None:
            continue
        cf = m2.get("cand_f")
        if cf is None or len(cf) == 0:
            continue
        cand_parts.append(_ten(cf, dev))
        base_parts.append(_ten(m2["cand_base"], dev))
        step_cand_off[i] = c_off
        step_cand_n[i] = len(cf)
        step_has_m2[i] = True
        c0 = c_off
        c_off += len(cf)
        for k, _r in enumerate(cf):
            cand_rows.append((i, k, c0 + k))
        if m2.get("draws"):
            has_draws[i] = True
        for d in m2.get("draws", []):
            rem = [int(r) for r in d.get("rem_ids", [])]
            idx = int(d.get("idx", rem[0] if rem else 0))
            if idx not in rem:
                raise ValueError(
                    f"m2 draw idx {idx} not in rem_ids {rem} (T2-B-GPU pack)")
            draw.append((i, c0 + idx, rem.index(idx), float(d.get("logp", 0.0)),
                         [c0 + r for r in rem]))
        pm = list(m2.get("pool_mask", []))
        if pm:
            pool_step.append((i, [c0 + int(r) for r in pm]))
    if not cand_parts:
        return None
    cand_all = torch.cat(cand_parts, 0)
    base_all = torch.cat(base_parts, 0)
    nd = len(draw)
    rem_max = max((len(d[4]) for d in draw), default=1)
    rem_pad = torch.zeros(nd, rem_max, dtype=torch.long, device=dev)
    rem_col = torch.zeros(nd, dtype=torch.long, device=dev)
    rem_m = torch.zeros(nd, dtype=torch.long, device=dev)
    draw_step = torch.zeros(nd, dtype=torch.long, device=dev)
    draw_id = torch.zeros(nd, dtype=torch.long, device=dev)
    draw_logp = torch.zeros(nd, dtype=torch.float32, device=dev)
    for j, (i, gid, col, lp, rows) in enumerate(draw):
        rem_pad[j, :len(rows)] = _ten(rows, dev, torch.long)
        rem_col[j] = col
        rem_m[j] = len(rows)
        draw_step[j] = i
        draw_id[j] = gid
        draw_logp[j] = lp

    # per-step gated draw counts + per-trajectory step counts (stems have per-
    # step inf2 flags for the STEP-level denominator -- a trajectory's steps
    # share the same gate, so count separately for exactness of means)
    dcount_step = torch.zeros(len(stems), dtype=torch.float32, device=dev)
    if nd:
        dcount_step.scatter_add_(0, draw_step, torch.ones(nd, device=dev))
    return {"cand_all": cand_all, "base_all": base_all,
            "step_cand_off": step_cand_off, "step_cand_n": step_cand_n,
            "step_has_m2": step_has_m2, "has_draws": has_draws,
            "cand_rows": cand_rows, "pool_step": pool_step,
            "rem_pad": rem_pad, "rem_col": rem_col, "rem_m": rem_m,
            "nd": nd, "draw_step": draw_step, "draw_id": draw_id,
            "draw_logp": draw_logp, "dcount_step": dcount_step}


def _m2_loss(m2rec, coll, clip_eps, lambda_m2, beta_m2, dev):
    """M2 surrogate + KL_m2 terms, exact canonical reduction (§19):
    per-draw clipped ratio x A2 -> per-step mean over gated draws ->
    per-trajectory mean -> mean over trajectories.  Returns m2_loss_term for
    the caller to add (already includes -lambda / +beta)."""
    temp_m2 = float(C.TO1_R13_TEMP_M2)
    alpha_m2 = float(C.TO1_R13_ALPHA_M2)
    net = m2rec["net"]
    logit_all = (m2rec["base_all"] + alpha_m2 * torch.tanh(
        net(m2rec["cand_all"]).squeeze(-1))) / temp_m2

    # per-draw logp_new = log_softmax over rem_ids at the chosen column
    # (rows gathered by index -- rem_pad holds global candidate row ids)
    dlg = logit_all[m2rec["rem_pad"]]                           # [nd, rem_max]
    msk = m2rec["rem_m"].reshape(-1, 1) > torch.arange(
        m2rec["rem_pad"].shape[1], device=dev).reshape(1, -1)
    lg_row = torch.where(msk, dlg, torch.full_like(dlg, float("-inf")))
    lse = torch.logsumexp(lg_row, dim=-1)
    val = torch.gather(lg_row, 1, m2rec["rem_col"].reshape(-1, 1)).squeeze(1)
    lp_new = val - lse
    ratio2 = torch.exp(lp_new - m2rec["draw_logp"])
    gated = coll["inf2"][m2rec["draw_step"]]
    A2 = coll["adv2"][m2rec["draw_step"]]
    obj2 = torch.min(ratio2 * A2, torch.clamp(ratio2, 1.0 - clip_eps,
                                              1.0 + clip_eps) * A2)
    # weights: per-draw (1 / step gated draw count) * (1 / traj step count) *
    #          (1 / N_s2)
    dcount = m2rec["dcount_step"][m2rec["draw_step"]]
    traj_of_draw = coll["traj"][m2rec["draw_step"]]
    gated_step = coll["inf2"] & m2rec["has_draws"]
    cnt2 = torch.zeros(coll["n_traj"], dtype=torch.float32, device=dev)
    cnt2.scatter_add_(0, coll["traj"][gated_step],
                      torch.ones(int(gated_step.sum()), device=dev))
    n_s2 = int((cnt2 > 0).sum())
    eps_term = torch.zeros((), dtype=torch.float32, device=dev)
    if gated.any() and n_s2 > 0:
        w = (gated / dcount.clamp(min=1.0) /
             cnt2[traj_of_draw].clamp(min=1.0) / float(n_s2))
        w = w.clamp(max=1e8)
        loss2 = -(obj2 * w).sum()
        eps_term = loss2
    # kl_m2: per-step KL(softmax(adapter / T) || softmax(base / T)) over the
    # pool_mask rows, mean over ALL steps that have a pool (unconditioned on
    # informativeness -- canonical appends whenever m2_rec is present)
    kl_list = []
    net_out = net(m2rec["cand_all"]).squeeze(-1)
    s_t_full = (m2rec["base_all"] + alpha_m2 * torch.tanh(net_out)) / temp_m2
    s_r_full = m2rec["base_all"] / temp_m2
    for (i, rows) in m2rec["pool_step"]:
        rt = torch.tensor(rows, dtype=torch.long, device=dev)
        p_t = torch.softmax(s_t_full[rt], dim=-1)
        p_r = torch.softmax(s_r_full[rt], dim=-1)
        kl_list.append((p_t * (torch.log(p_t) - torch.log(p_r))).sum()
                       .reshape(1))
    m2_loss_term = torch.zeros((), dtype=torch.float32, device=dev)
    m2_kl = 0.0
    if kl_list:
        kl_m2v = torch.cat(kl_list).mean()
        m2_kl = float(kl_m2v.detach())
        m2_loss_term = m2_loss_term + beta_m2 * kl_m2v
    # surrogate term: -lambda_m2 * mean over trajectories
    m2_loss_term = m2_loss_term + lambda_m2 * eps_term
    return {"m2_loss_term": m2_loss_term, "n_s2_trajs": int(n_s2),
            "kl_m2": float(m2_kl), "n_pool_steps": len(kl_list)}


# ---------------------------------------------------------------------------
# success gate (§31 -- A requires greedy-up AND gap-narrow AND bestofN stable)
# ---------------------------------------------------------------------------
def gpu_success_gate(e0, final, margin, reduce, bestn_tol=0.10):
    """T2-B double-gate + best-of-N no-regression as an explicit third gate.
    e0/final are `t2a_split_summary` dicts.
    Returns (ok, gates{dict}, code)."""
    g1 = bool(final.get("greedy_total", 0.0)
              > e0.get("greedy_total", 0.0) + float(margin))
    e0_gap = float(e0.get("bestofN_gain", 0.0))
    f_gap = float(final.get("bestofN_gain", 0.0))
    g2 = bool(e0_gap > 0.0 and f_gap <= float(reduce) * e0_gap)
    e0_bn = float(e0.get("bestN_total", 0.0))
    f_bn = float(final.get("bestN_total", 0.0))
    g3 = bool(e0_bn > 0 and f_bn >= (1.0 - bestn_tol) * e0_bn)
    ok = bool(g1 and g2 and g3)
    if ok:
        code = "A"
    elif g1 and g2:
        code = "B+gap"
    elif g1 or g2:
        code = "B"
    else:
        code = "C"
    return ok, {"greedy_up": g1, "gap_narrow": g2,
                "bestn_no_regress": g3}, code


# ---------------------------------------------------------------------------
# GPU preflight (§12/§13) + AMP probe (§20-22)
# ---------------------------------------------------------------------------
def _cpu_cores():
    logical = os.cpu_count() or 1
    try:
        with open("/proc/cpuinfo") as f:
            ids = [ln.split(":")[1].strip() for ln in f
                   if ln.startswith("core id")]
        physical = len(set(ids)) if ids else logical
    except OSError:
        physical = logical
    return {"physical": int(physical), "logical": int(logical)}


def _ram_gb():
    try:
        import resource
        used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    except (ImportError, AttributeError):
        used = 0.0
    total = 0.0
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        total = pages * size / 1024.0**3
    except (AttributeError, ValueError, OSError):
        try:
            with open("/proc/meminfo") as f:
                for ln in f:
                    if ln.startswith("MemTotal"):
                        total = float(ln.split()[1]) / 1024.0**2
                        break
        except OSError:
            total = 0.0
    return {"total_gb": round(total, 1), "used_gb": round(used, 1)}


def gpu_preflight(device="auto") -> dict[str, Any]:
    """Environment + GPU snapshot (§13)."""
    import platform
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
    }
    info.update(_cpu_cores())
    info.update(_ram_gb())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        try:
            free, tot = torch.cuda.mem_get_info(0)
        except (RuntimeError, OSError):
            free = tot = props.total_memory
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = props.name
        info["gpu_cc"] = f"{props.major}.{props.minor}"
        info["vram_total_gb"] = round(tot / 1024.0**3, 2)
        info["vram_free_gb"] = round(free / 1024.0**3, 2)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def probe_amp(device, m3g, F, evid, sf_ps, iters=4) -> str:
    """Micro-benchmark the packed M3 residual forward on the real first-batch
    data across fp32 / bf16 (autocast) / fp16.  Returns the faster *stable*
    mode (finite grads and loss within 1e-2 of the fp32 reference).  CPU -> fp32.
    §20-22: auto-detect, record; the update itself stays fp32 for parity v1."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return "fp32"
    modes = ["fp32"]
    if torch.cuda.is_bf16_supported():
        modes.append("bf16")
    modes.append("fp16")
    F_ev = torch.cat([F, evid], dim=-1) if evid is not None else F
    best, best_t = "fp32", 1e18
    for mode in modes:
        try:
            t0 = time.time()
            finite = True
            tol_ok = True
            ref = None
            for _ in range(iters):
                m3g.zero_grad(set_to_none=True)
                if mode == "fp32":
                    d = m3g.alpha_prop * torch.tanh(
                        m3g.resid_prop(F_ev).squeeze(-1))
                elif mode == "bf16":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        d = m3g.alpha_prop * torch.tanh(
                            m3g.resid_prop(F_ev).squeeze(-1))
                    d = d.float()
                else:
                    with torch.autocast("cuda", dtype=torch.float16):
                        d = m3g.alpha_prop * torch.tanh(
                            m3g.resid_prop(F_ev).squeeze(-1))
                    d = d.float()
                loss = d.sum()
                loss.backward()
                g = m3g.resid_prop.weight.grad
                finite &= bool(torch.isfinite(loss).item()
                               and (g is None or torch.isfinite(g).all().item()))
                if ref is None:
                    ref = float(loss.detach())
                else:
                    tol_ok &= abs(float(loss.detach()) - ref) < 1e-2 * max(
                        1.0, abs(ref))
            el = time.time() - t0
            if finite and tol_ok and el < best_t:
                best, best_t = mode, el
        except Exception:   # noqa: BLE001 - unsupported path -> keep fp32
            continue
    return best


# ---------------------------------------------------------------------------
# run-identity / checkpoint / resume (§45-47) -- pure helpers shared with the
# runner so tests exercise the exact code path without importing the CLI.
# ---------------------------------------------------------------------------
def cfg_hash(cfg) -> str:
    """Stable sha-256[:12] over the resolved JSON config (checkpoint identity)."""
    from causal_schedule_lab.m3.t2b_gpu_monitor import _jsonable
    raw = json.dumps(_jsonable(cfg.to_dict()), sort_keys=True,
                     default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def assert_no_held_train_overlap(train_iids: set, held_pairs) -> None:
    """Fail fast when any held/val pair instance id appears in train (§10/§41).

    `held_pairs` = iterable of (iid, eid) -- the split's held (val) pairs."""
    overlap = sorted(i for (i, _e) in held_pairs if i in train_iids)
    if overlap:
        raise ValueError(
            "held/val iid appears in train (train-source filter bug): "
            f"{overlap} -- T2-B-GPU NEVER trains on the held ruler")


def save_policy_ckpt(path, payload: dict[str, Any]) -> str:
    """Atomic torch.save of the per-cycle checkpoint payload."""
    path = os.fspath(path)
    tmp = f"{path}.tmp.{os.getpid()}"
    tmpd = os.path.dirname(tmp) or "."
    os.makedirs(tmpd, exist_ok=True)
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def _find_snapshot_loader(jpol):
    """Return the policy-level snapshot loader used by the canonical loop
    (JointAgenticPolicy.load_snapshot), tolerating test doubles without it."""
    ld = getattr(jpol, "load_snapshot", None)
    if ld is not None:
        return ld
    for m in (getattr(jpol, "m3", None), getattr(jpol, "m2", None)):
        ld = getattr(m, "load_snapshot", None)
        if ld is not None:
            return ld
    raise NotImplementedError(
        "resume requires a policy snapshot loader (load_snapshot)")


def save_policy_state(jpol, r6_sel=None) -> dict[str, Any]:
    """Snapshot the live CPU policy (m3 + m2 + r6) as a saveable dict."""
    out: dict[str, Any] = {}
    snap = None
    if hasattr(jpol, "snapshot"):
        snap = jpol.snapshot()                 # {"m3": ..., "m2": ...}
    else:
        m3 = getattr(jpol, "m3", None)
        m2 = getattr(jpol, "m2", None)
        if m3 is not None and hasattr(m3, "snapshot"):
            snap = {"m3": m3.snapshot()}
        if m2 is not None and hasattr(m2, "snapshot"):
            snap = {"m2": m2.snapshot()} if snap is None else {**snap,
                                                               "m2": m2.snapshot()}
    if snap:
        out.update(snap)
    if r6_sel is not None:
        sd = getattr(r6_sel, "state_dict", None)
        if sd is not None:
            out["r6"] = sd()
    return out


def load_policy_state(pol: dict[str, Any], jpol, r6_sel=None) -> None:
    """Apply a `save_policy_state` payload to the live policy (both halves +
    the optional frozen r6 selector).  Raises on an absent loader (fail fast)."""
    if hasattr(jpol, "load_snapshot"):
        if "m3" in pol or "m2" in pol:
            jpol.load_snapshot(pol)
    else:
        if "m3" in pol:
            _find_snapshot_loader(getattr(jpol, "m3", jpol))(pol["m3"])
        if "m2" in pol:
            _find_snapshot_loader(getattr(jpol, "m2", jpol))(pol["m2"])
    if "r6" in pol and r6_sel is not None:
        ld = getattr(r6_sel, "load_state_dict", None)
        if ld is not None:
            ld(pol["r6"])


def resume_from_ckpt(path, cfg, jpol, r6_sel=None) -> dict[str, Any]:
    """Load a T2-B-GPU checkpoint and apply it to the live policy.

    Restores weights (m3 + m2), the R6 selector, cycle, RNG state, wall-clock;
    verifies the config hash matches the checkpoint identity (fail fast).
    Returns dict{cycle, rng_state, wall_s, meta, loaded}."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck.get("meta", {})
    if meta.get("config_hash") and cfg_hash(cfg) != meta["config_hash"]:
        raise ValueError(
            f"resume {path}: config hash {meta.get('config_hash')} != current "
            f"{cfg_hash(cfg)} -- resume requires the identical resolved config")
    pol = ck.get("policy") or (ck.get("state") or {}).get("policy")
    if not isinstance(pol, dict):
        # legacy full-object payload (jpol pickled directly)
        ld = _find_snapshot_loader(jpol)
        ld(pol)
    elif pol:
        load_policy_state(pol, jpol, r6_sel)
    r6_st = ck.get("r6_state") or ((ck.get("state") or {}).get("r6_anchor"))
    if r6_st is not None and r6_sel is not None:
        ld = getattr(r6_sel, "load_state_dict", None)
        if ld is not None:
            try:
                ld(r6_st)
            except Exception:            # stale/foreign keys -> fail fast
                raise ValueError(
                    f"resume {path}: r6_state does not match the live selector")
    return {"cycle": int(meta.get("cycle", 0)),
            "rng_state": ck.get("rng_state"),
            "wall_s": float(meta.get("wall_s", 0.0)),
            "meta": meta, "loaded": True}


def make_ckpt_payload(jpol, r6_sel, scorer, cfg, cycle, rng_state, wall_s,
                      promoted: bool = False) -> dict[str, Any]:
    """The canonical T2-B-GPU checkpoint payload (trainer-owned schema):
    `policy` = save_policy_state dict, `meta` carries identity + provenance."""
    pol = save_policy_state(jpol, r6_sel)
    return {
        "policy": pol,
        "r6_state": pol.get("r6"),
        "meta": {
            "phase": "t2b_gpu_multipath_joint_grpo",
            "runtime": "R20_lexicographic", "reward": "stagewise_A2_A3",
            "best_of_n": "post_hoc_diagnostic_only",
            "extraction_gap": "difference_bestofN_minus_greedy",
            "formal_test_access": 0, "identified": False,
            "promoted": bool(promoted), "cycle": int(cycle),
            "config_hash": cfg_hash(cfg), "config": cfg.to_dict(),
            "wall_s": round(float(wall_s), 1),
        },
        "rng_state": rng_state,
    }