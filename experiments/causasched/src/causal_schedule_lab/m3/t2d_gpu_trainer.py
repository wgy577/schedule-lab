"""GPU update bridge for T2-D structured actors.

CPU keeps environment construction and FixedDecisionReplay.  Only collected policy
tensors, the two actors, log-prob/KL computation, loss and backward move to CUDA.
"""
from __future__ import annotations

import copy
import math
import threading
import time
import weakref

import torch
from torch import nn

from . import joint_grpo as JG

_CANONICAL_UPDATE = JG.grpo_update_joint
_DEVICE = "auto"
_GPU_POLICIES = weakref.WeakKeyDictionary()
_M3_MICROBATCH_ROWS = 1024
_M2_MICROBATCH_ROWS = 1536


def set_runtime(device="auto"):
    global _DEVICE
    _DEVICE = str(device)


def set_microbatch_rows(m3_rows=1024, m2_rows=1536):
    """Tune CUDA work size without changing the outer GRPO update batch."""
    global _M3_MICROBATCH_ROWS, _M2_MICROBATCH_ROWS
    _M3_MICROBATCH_ROWS = max(1, int(m3_rows))
    _M2_MICROBATCH_ROWS = max(1, int(m2_rows))


def _device():
    if _DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("T2-D GPU runner requested CUDA but CUDA is unavailable")
    return torch.device(_DEVICE)


def _move_groups(groups, dev):
    """Pack variable sets by field so each field takes one H2D transfer."""
    moved = []
    records = []
    for group in groups:
        g = dict(group)
        g["trajs"] = []
        for trajectory in group.get("trajs", []):
            tr = dict(trajectory)
            tr["steps"] = []
            for record in trajectory.get("steps", []):
                rec = dict(record)
                if rec.get("m2_rec") is not None:
                    rec["m2_rec"] = dict(rec["m2_rec"])
                tr["steps"].append(rec)
                records.append(rec)
            g["trajs"].append(tr)
        moved.append(g)

    def pack_variable(owners, key):
        selected = [(owner, owner.get(key)) for owner in owners
                    if torch.is_tensor(owner.get(key))]
        if not selected:
            return
        lengths = [int(value.shape[0]) for _owner, value in selected]
        packed = torch.cat([value for _owner, value in selected], dim=0).to(
            dev, non_blocking=True)
        offset = 0
        for (owner, _value), length in zip(selected, lengths):
            owner[key] = packed[offset:offset + length]
            offset += length

    def pack_fixed(owners, key):
        selected = [(owner, owner.get(key)) for owner in owners
                    if torch.is_tensor(owner.get(key))]
        if not selected:
            return
        packed = torch.stack([value for _owner, value in selected], dim=0).to(
            dev, non_blocking=True)
        for idx, (owner, _value) in enumerate(selected):
            owner[key] = packed[idx]

    for key in ("F_pool", "logits_old", "evid"):
        pack_variable(records, key)
    for key in ("sf_t", "pool_stats", "traj_ctx"):
        pack_fixed(records, key)
    m2_records = [rec["m2_rec"] for rec in records if rec.get("m2_rec")]
    for key in ("cand_f", "cand_base", "cand_latent", "cand_memory", "enab_mask",
                "cand_app_raw", "cand_app_mask", "cand_min_hop",
                "cand_support_count"):
        pack_variable(m2_records, key)
    # One transfer per shared decision graph.
    trace_cache = {}
    for rec in m2_records:
        graph = rec.get("cand_trace_graph")
        if graph is not None:
            if id(graph) not in trace_cache:
                trace_cache[id(graph)] = {k: v.to(dev) for k, v in graph.items()}
            rec["cand_trace_graph"] = trace_cache[id(graph)]
    pack_fixed(m2_records, "state_context")
    return moved


def _gpu_policy(jpol, dev):
    # Never attach the CUDA copy to jpol: jpol is pickled into CPU rollout workers.
    # Keeping CUDA state on it would serialize a duplicate model for every job.
    gpu = _GPU_POLICIES.get(jpol)
    if gpu is None or next(gpu.m3.parameters()).device != dev:
        gpu = copy.deepcopy(jpol)
        # Optimizers are runtime state, not part of the copied policy contract.
        gpu.__dict__.pop("_t2d_optimizer", None)
        gpu.__dict__.pop("_t2d_optimizer_param_ids", None)
        gpu.m2.to(dev)
        gpu.m3.to(dev)
        # A CPU-loaded checkpoint carries this transient state into the GPU copy;
        # canonical GRPO consumes it exactly once when constructing AdamW.
        if hasattr(jpol, "_t2d_resume_optimizer_state"):
            delattr(jpol, "_t2d_resume_optimizer_state")
        _GPU_POLICIES[jpol] = gpu
    # Synchronize a possible best-snapshot rollback while retaining Adam moments.
    with torch.no_grad():
        for src, dst in zip(jpol.m2.parameters(), gpu.m2.parameters()):
            dst.copy_(src.detach().to(dev))
        for src, dst in zip(jpol.m3.parameters(), gpu.m3.parameters()):
            dst.copy_(src.detach().to(dev))
    return gpu


def optimizer_state_cpu(jpol):
    """Export live AdamW state while keeping the CUDA policy out of checkpoints."""
    gpu = _GPU_POLICIES.get(jpol)
    opt = getattr(gpu, "_t2d_optimizer", None) if gpu is not None else None
    if opt is None:
        opt = getattr(jpol, "_t2d_optimizer", None)
    if opt is None:
        return None

    def cpu_copy(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: cpu_copy(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu_copy(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cpu_copy(v) for v in value)
        return value
    return cpu_copy(opt.state_dict())


def _copy_back(gpu, cpu):
    with torch.no_grad():
        for src, dst in zip(gpu.m2.parameters(), cpu.m2.parameters()):
            if src.requires_grad:
                dst.copy_(src.detach().cpu())
        for src, dst in zip(gpu.m3.parameters(), cpu.m3.parameters()):
            if src.requires_grad:
                dst.copy_(src.detach().cpu())


def _gpu_utilization_sample(dev):
    if dev.type != "cuda" or not hasattr(torch.cuda, "utilization"):
        return None
    try:
        return float(torch.cuda.utilization(dev))
    except Exception:  # NVML is optional on some AutoDL images
        return None


def _bucket_size(n):
    """Tight power-of-two buckets limit padding without returning to batch=1."""
    n = max(int(n), 1)
    return 1 << int(math.ceil(math.log2(n)))


def _pad_rows(rows, value=0.0):
    return nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=value)


def _optimizer(jpol, params_m2, params_m3, lr_m2, lr_m3,
               optimizer_persistent):
    params = params_m3 + params_m2
    opt = getattr(jpol, "_t2d_optimizer", None) if optimizer_persistent else None
    current_ids = tuple(id(p) for p in params)
    if opt is None or getattr(jpol, "_t2d_optimizer_param_ids", ()) != current_ids:
        if optimizer_persistent:
            groups = []
            if params_m2:
                groups.append({"params": params_m2, "lr": lr_m2, "actor": "m2"})
            if params_m3:
                groups.append({"params": params_m3, "lr": lr_m3, "actor": "m3"})
            opt = torch.optim.AdamW(groups, weight_decay=0.0)
            jpol._t2d_optimizer = opt
            jpol._t2d_optimizer_param_ids = current_ids
            resume = getattr(jpol, "_t2d_resume_optimizer_state", None)
            if resume is not None:
                try:
                    opt.load_state_dict(resume)
                except (ValueError, RuntimeError, KeyError) as exc:
                    # Later variants add trainable root/appearance propagation
                    # parameters. An older AdamW state may have a different
                    # parameter layout.  Actor weights have already migrated via
                    # the name-based policy snapshot loader; safely restart only
                    # AdamW instead of rejecting a useful checkpoint.
                    print("[t2m] optimizer migration: actor snapshot restored; "
                          f"old AdamW state was reinitialized ({type(exc).__name__})",
                          flush=True)
                delattr(jpol, "_t2d_resume_optimizer_state")
        else:
            opt = torch.optim.AdamW(params, lr=lr_m3, weight_decay=0.0)
    return opt


def _batched_m3(jpol, records):
    """Run variable proposal sets in padded size buckets; return graph-bearing logits."""
    result = {}
    buckets = {}
    for rec in records:
        buckets.setdefault(_bucket_size(len(rec["F_pool"])), []).append(rec)
    for _width, rows in sorted(buckets.items()):
        lengths = [len(r["F_pool"]) for r in rows]
        F = _pad_rows([r["F_pool"] for r in rows])
        sf = torch.stack([r["sf_t"].reshape(-1) for r in rows])
        ps = torch.stack([r["pool_stats"].reshape(-1) for r in rows])
        tc = torch.stack([
            r["traj_ctx"].reshape(-1) if r.get("traj_ctx") is not None else
            r["sf_t"].new_zeros(jpol.m3.trajectory_dim) for r in rows])
        if jpol.m3.evidence_dim:
            ev_rows = [r.get("evid") if r.get("evid") is not None else
                       r["F_pool"].new_zeros((len(r["F_pool"]), jpol.m3.evidence_dim))
                       for r in rows]
            ev = _pad_rows(ev_rows)
        else:
            ev = F.new_zeros((len(rows), F.shape[1], 0))
        lens = torch.as_tensor(lengths, device=F.device)
        mask = torch.arange(F.shape[1], device=F.device).unsqueeze(0) < lens.unsqueeze(1)
        pair_mask = _pad_rows([
            (r["pair_mask"].to(dtype=torch.bool)
             if r.get("pair_mask") is not None else
             torch.zeros(len(r["F_pool"]), dtype=torch.bool))
            for r in rows], value=False).to(F.device)
        prop, stop, base_prop, base_stop = jpol.m3.action_and_base_logits_batched(
            F, sf, ps, ev, tc, mask, pair_mask)
        for i, (rec, length) in enumerate(zip(rows, lengths)):
            result[id(rec)] = (
                torch.cat([prop[i, :length], stop[i:i + 1]]),
                torch.cat([base_prop[i, :length], base_stop[i:i + 1]]))
    return result


def _batched_m2(jpol, records, temperature):
    """Batch every changing remaining-root set used by Plackett-Luce draws/KL."""
    queries = []
    zero_kl = {}
    for rec in records:
        m2 = rec.get("m2_rec")
        if m2 is None:
            continue
        for draw_index, draw in enumerate(m2["draws"]):
            ids = list(draw["rem_ids"])
            queries.append((rec, "draw", draw_index, ids,
                            ids.index(draw["idx"])))
        ids = list(m2["pool_mask"])
        if ids:
            queries.append((rec, "kl", 0, ids, None))
        else:
            zero_kl[id(rec)] = m2["cand_base"].new_zeros(())
    output = {}
    buckets = {}
    for q in queries:
        buckets.setdefault(_bucket_size(len(q[3])), []).append(q)
    for _width, rows in sorted(buckets.items()):
        lat, loc, bas, mem, apps, app_masks, states, lengths = [], [], [], [], [], [], [], []
        for rec, _kind, _index, ids, _pos in rows:
            m2 = rec["m2_rec"]
            idx = torch.as_tensor(ids, dtype=torch.long, device=m2["cand_base"].device)
            lat.append(m2["cand_latent"][idx])
            loc.append(m2["cand_f"][idx])
            bas.append(m2["cand_base"][idx])
            memory = m2.get("cand_memory")
            mem.append(memory[idx] if memory is not None else
                       m2["cand_f"].new_zeros((len(ids), jpol.m2.memory_dim)))
            if m2.get("cand_app_raw") is not None:
                apps.append(m2["cand_app_raw"][idx])
                app_masks.append(m2["cand_app_mask"][idx])
            state = m2.get("state_context")
            states.append(state.reshape(-1) if state is not None else
                          m2["cand_f"].new_zeros(jpol.m2.state_dim))
            lengths.append(len(ids))
        p_lat, p_loc, p_bas, p_mem = map(_pad_rows, (lat, loc, bas, mem))
        p_app = _pad_rows(apps) if apps else None
        p_app_mask = _pad_rows(app_masks, value=False) if app_masks else None
        lens = torch.as_tensor(lengths, device=p_bas.device)
        mask = torch.arange(p_bas.shape[1], device=p_bas.device).unsqueeze(0) < lens.unsqueeze(1)
        scores = jpol.m2.final_logits_batched(
            p_lat, p_loc, p_bas, torch.stack(states), p_mem, mask,
            p_app, p_app_mask,
            [(rec["m2_rec"].get("cand_trace_graph"), ids)
             for rec, _kind, _index, ids, _pos in rows])
        for i, ((rec, kind, index, _ids, pos), length) in enumerate(zip(rows, lengths)):
            score = scores[i, :length] / temperature
            eps2 = float(JG.C.TO1_R13_MIX_EPS)
            probs = torch.softmax(score, dim=0)
            probs = (1.0 - eps2) * probs + eps2 / max(length, 1)
            if kind == "draw":
                output[(id(rec), "draw", index)] = torch.log(
                    probs[pos].clamp_min(1e-12))
            else:
                ref = p_bas[i, :length] / temperature
                pt, pr = probs, torch.softmax(ref, dim=0)
                pr = (1.0 - eps2) * pr + eps2 / max(length, 1)
                output[(id(rec), "kl", 0)] = (pt * (torch.log(pt) - torch.log(pr))).sum()
    for rec_id, value in zero_kl.items():
        output[(rec_id, "kl", 0)] = value
    return output


def _mixture_probs(logits, valid, temperature, eps):
    # Mask before softmax.  T2-F disables learned STOP while executable
    # proposals exist; padded proposal columns were already invalid, but the
    # former implementation never needed to mask the real STOP column.
    masked = logits.masked_fill(~valid, -torch.inf)
    probs = torch.softmax(masked / temperature, dim=1)
    count = valid.sum(dim=1, keepdim=True).to(logits.dtype)
    return ((1.0 - eps) * probs + eps * valid.to(logits.dtype) / count).clamp_min(1e-12)


def _vectorized_m3_terms(jpol, records, weights, T, eps, clip_eps, beta,
                         microbatch_rows=None):
    microbatch_rows = int(microbatch_rows or _M3_MICROBATCH_ROWS)
    dev = next(jpol.m3.parameters()).device
    zero = torch.zeros((), dtype=torch.float32, device=dev)
    sur_total, kl_ref_total, kl_old_total, clip_total, entropy_total = (
        zero, zero.clone(), zero.clone(), zero.clone(), zero.clone())
    buckets = {}
    for rec in records:
        buckets.setdefault(_bucket_size(len(rec["F_pool"])), []).append(rec)
    work = []
    for _width, bucket_rows in sorted(buckets.items()):
        work.extend(bucket_rows[p:p + int(microbatch_rows)]
                    for p in range(0, len(bucket_rows), int(microbatch_rows)))
    denom = max(len(records), 1)
    for rows in work:
        lengths = [len(r["F_pool"]) for r in rows]
        # Keep the full 48xK rollout batch in host RAM and stream only this
        # autograd microbatch to CUDA.  Moving every variable proposal tensor to
        # CUDA up front retained tens of GiB while also building the backward
        # graph and was the main update-time memory spike.
        F = _pad_rows([r["F_pool"] for r in rows]).to(dev)
        sf = torch.stack([r["sf_t"].reshape(-1) for r in rows]).to(dev)
        ps = torch.stack([r["pool_stats"].reshape(-1) for r in rows]).to(dev)
        tc = torch.stack([(r["traj_ctx"].reshape(-1) if r.get("traj_ctx") is not None else
                           r["sf_t"].new_zeros(jpol.m3.trajectory_dim))
                          for r in rows]).to(dev)
        if jpol.m3.evidence_dim:
            ev = _pad_rows([r["evid"] if r.get("evid") is not None else
                            r["F_pool"].new_zeros((len(r["F_pool"]),
                                                  jpol.m3.evidence_dim))
                            for r in rows]).to(dev)
        else:
            ev = F.new_zeros((len(rows), F.shape[1], 0))
        lens = torch.as_tensor(lengths, device=F.device)
        prop_valid = torch.arange(F.shape[1], device=F.device)[None] < lens[:, None]
        pair_mask = _pad_rows([
            (r["pair_mask"].to(dtype=torch.bool)
             if r.get("pair_mask") is not None else
             torch.zeros(len(r["F_pool"]), dtype=torch.bool))
            for r in rows], value=False).to(dev)
        prop, stop, base_prop, base_stop = jpol.m3.action_and_base_logits_batched(
            F, sf, ps, ev, tc, prop_valid, pair_mask)
        logits = torch.cat([prop, stop[:, None]], dim=1)
        base = torch.cat([base_prop, base_stop[:, None]], dim=1)
        stop_valid = torch.as_tensor(
            [bool(r.get("stop_allowed", True)) for r in rows],
            dtype=torch.bool, device=F.device)[:, None]
        valid = torch.cat([prop_valid, stop_valid], dim=1)
        old_prop = _pad_rows([r["logits_old"][:-1] for r in rows],
                             value=-torch.inf).to(dev)
        old_stop = torch.stack([r["logits_old"][-1] for r in rows]).to(dev)
        old = torch.cat([old_prop, old_stop[:, None]],
                        dim=1)
        p, pref, pold = (_mixture_probs(x, valid, T, eps) for x in (logits, base, old))
        actions = torch.as_tensor([F.shape[1] if int(r["a"]) == n else int(r["a"])
                                   for r, n in zip(rows, lengths)], device=F.device)
        lp_new = torch.log(p.gather(1, actions[:, None]).squeeze(1))
        old_lp = torch.as_tensor([float(r["logp_old"]) for r in rows],
                                 dtype=lp_new.dtype, device=F.device)
        ratio = torch.exp(lp_new - old_lp)
        advantage = torch.as_tensor([float(r["adv3"]) for r in rows],
                                    dtype=ratio.dtype, device=F.device)
        weight = torch.as_tensor([weights[id(r)][0] for r in rows],
                                 dtype=ratio.dtype, device=F.device)
        surrogate = torch.minimum(ratio * advantage,
                                  ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantage)
        sur_chunk = (surrogate * weight).sum()
        kl_ref_chunk = (p * (torch.log(p) - torch.log(pref))).masked_fill(
            ~valid, 0.0).sum()
        kl_old_chunk = (p * (torch.log(p) - torch.log(pold))).masked_fill(
            ~valid, 0.0).sum()
        entropy_chunk = -(p * torch.log(p)).masked_fill(~valid, 0.0).sum()
        (-sur_chunk + float(beta) * kl_ref_chunk / denom).backward()
        sur_total += sur_chunk.detach()
        kl_ref_total += kl_ref_chunk.detach()
        kl_old_total += kl_old_chunk.detach()
        entropy_total += entropy_chunk.detach()
        clip_total += (((ratio > 1.0 + clip_eps) | (ratio < 1.0 - clip_eps)) &
                       (weight > 0)).sum().detach()
    return (sur_total, kl_ref_total / denom, kl_old_total / denom, clip_total,
            entropy_total / denom)


def _vectorized_m2_terms(jpol, records, weights, temperature, clip_eps,
                         lambda_m2, beta, microbatch_rows=None):
    microbatch_rows = int(microbatch_rows or _M2_MICROBATCH_ROWS)
    dev = next(jpol.m2.parameters()).device
    zero = torch.zeros((), dtype=torch.float32, device=dev)
    sur_total, kl_total = zero, zero.clone()
    queries, n_m2 = [], 0
    for rec in records:
        m2 = rec.get("m2_rec")
        if m2 is None:
            continue
        n_m2 += 1
        for draw in m2["draws"]:
            ids = list(draw["rem_ids"])
            queries.append((rec, draw, ids, ids.index(draw["idx"]), False))
        ids = list(m2["pool_mask"])
        if ids:
            queries.append((rec, None, ids, 0, True))
    buckets = {}
    for q in queries:
        buckets.setdefault(_bucket_size(len(q[2])), []).append(q)
    work = []
    for _width, bucket_rows in sorted(buckets.items()):
        work.extend(bucket_rows[p:p + int(microbatch_rows)]
                    for p in range(0, len(bucket_rows), int(microbatch_rows)))
    for rows in work:
        lat, loc, bas, mem, apps, app_masks, states, lengths = [], [], [], [], [], [], [], []
        for rec, _draw, ids, _pos, _is_kl in rows:
            m2 = rec["m2_rec"]
            idx = torch.as_tensor(ids, dtype=torch.long, device=m2["cand_base"].device)
            lat.append(m2["cand_latent"][idx]); loc.append(m2["cand_f"][idx])
            bas.append(m2["cand_base"][idx])
            memory = m2.get("cand_memory")
            mem.append(memory[idx] if memory is not None else
                       m2["cand_f"].new_zeros((len(ids), jpol.m2.memory_dim)))
            if m2.get("cand_app_raw") is not None:
                apps.append(m2["cand_app_raw"][idx])
                app_masks.append(m2["cand_app_mask"][idx])
            state = m2.get("state_context")
            states.append(state.reshape(-1) if state is not None else
                          m2["cand_f"].new_zeros(jpol.m2.state_dim))
            lengths.append(len(ids))
        p_lat, p_loc, p_bas, p_mem = (
            _pad_rows(values).to(dev) for values in (lat, loc, bas, mem))
        p_app = _pad_rows(apps).to(dev) if apps else None
        p_app_mask = (_pad_rows(app_masks, value=False).to(dev)
                      if app_masks else None)
        states_t = torch.stack(states).to(dev)
        lens = torch.as_tensor(lengths, device=p_bas.device)
        valid = torch.arange(p_bas.shape[1], device=p_bas.device)[None] < lens[:, None]
        score = jpol.m2.final_logits_batched(
            p_lat, p_loc, p_bas, states_t, p_mem, valid,
            p_app, p_app_mask,
            [(rec["m2_rec"].get("cand_trace_graph"), ids)
             for rec, _draw, ids, _pos, _is_kl in rows]) / temperature
        eps2 = float(JG.C.TO1_R13_MIX_EPS)
        probs = torch.softmax(score, dim=1)
        count = valid.sum(dim=1, keepdim=True).to(score.dtype)
        probs = ((1.0 - eps2) * probs +
                 eps2 * valid.to(score.dtype) / count).clamp_min(1e-12)
        logp = torch.log(probs)
        positions = torch.as_tensor([q[3] for q in rows], device=score.device)
        selected = logp.gather(1, positions[:, None]).squeeze(1)
        draw_mask = torch.as_tensor([not q[4] for q in rows], dtype=torch.bool,
                                    device=score.device)
        old_lp = torch.as_tensor([float(q[1]["logp"]) if q[1] is not None else 0.0
                                  for q in rows], dtype=score.dtype, device=score.device)
        ratio = torch.exp(selected - old_lp)
        advantage = torch.as_tensor([float(q[0]["adv2"]) for q in rows],
                                    dtype=score.dtype, device=score.device)
        weight = torch.as_tensor([weights[id(q[0])][1] if not q[4] else 0.0
                                  for q in rows], dtype=score.dtype, device=score.device)
        surrogate = torch.minimum(ratio * advantage,
                                  ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantage)
        sur_chunk = (surrogate * weight * draw_mask).sum()
        ref_score = (p_bas / temperature).masked_fill(~valid, -torch.inf)
        pt, pr = probs, torch.softmax(ref_score, dim=1)
        pr = ((1.0 - eps2) * pr +
              eps2 * valid.to(score.dtype) / count).clamp_min(1e-12)
        kl_vec = (pt * (torch.log(pt.clamp_min(1e-12)) -
                        torch.log(pr.clamp_min(1e-12)))).masked_fill(~valid, 0.0).sum(dim=1)
        kl_mask = ~draw_mask
        kl_chunk = (kl_vec * kl_mask).sum()
        (-float(lambda_m2) * sur_chunk + float(beta) * kl_chunk /
         max(n_m2, 1)).backward()
        sur_total += sur_chunk.detach()
        kl_total += kl_chunk.detach()
    return sur_total, kl_total / max(n_m2, 1), n_m2


def _batched_update(jpol, groups, stage, T, eps, clip_eps, seeds, epochs,
                    optimizer_persistent):
    """Stagewise GRPO with batched actors and fully vectorized loss reduction."""
    train_m2, train_m3 = stage in ("A", "C"), stage in ("B", "C")
    bundles, n_steps, records = [], 0, []
    for group in groups:
        inf, inf2 = bool(group.get("informative")), bool(group.get("info2", group.get("informative")))
        for tr in group.get("trajs", []):
            sts = tr["steps"]; n_steps += len(sts)
            for rec in sts:
                rec.setdefault("inf3", inf); rec.setdefault("inf2", inf2)
                rec.setdefault("adv3", 0.0); rec.setdefault("adv2", 0.0)
                records.append(rec)
            bundles.append((sts, inf, inf2))
    params_m3 = [p for p in jpol.m3.parameters() if p.requires_grad] if train_m3 else []
    params_m2 = [p for p in jpol.m2.parameters() if p.requires_grad] if train_m2 else []
    params = params_m3 + params_m2
    if not params:
        return {"n_informative_trajectories": 0, "n_trajectories": len(bundles),
                "n_steps": n_steps, "epochs": [], "reason": "no_trainable_params",
                "n_informative_trajectories_m2": 0, "final_kl_m3": None, "final_kl_m2": None}
    opt = _optimizer(jpol, params_m2, params_m3,
                     float(JG.C.TO1_R13_LR_M2 if train_m2 else 0.0),
                     float(JG.C.TO1_R13_LR_M3 if train_m3 else 0.0), optimizer_persistent)
    torch.manual_seed(0 if seeds is None else int(seeds))
    n_info_traj = sum(1 for sts, inf, _ in bundles if inf and sts)
    count_inf2 = sum(1 for sts, _, inf2 in bundles if inf2 and sts)
    weights, t3, t2 = {}, 0, 0
    for sts, _inf, _inf2 in bundles:
        r3 = [r for r in sts if bool(r["inf3"])]
        r2 = [r for r in sts if bool(r["inf2"]) and r.get("m2_rec") is not None
              and r["m2_rec"].get("draws")]
        t3 += bool(r3); t2 += bool(r2)
        for r in sts:
            weights[id(r)] = [0.0, 0.0]
        for r in r3:
            weights[id(r)][0] = 1.0 / len(r3)
        for r in r2:
            weights[id(r)][1] = 1.0 / (len(r2) * len(r["m2_rec"]["draws"]))
    for value in weights.values():
        value[0] /= max(t3, 1); value[1] /= max(t2, 1)
    per_epoch = []
    for ep in range(epochs):
        if not ((train_m3 and t3) or (train_m2 and t2)):
            per_epoch.append({"epoch": ep, "loss": 0.0,
                              "n_informative_trajectories": 0, "n_steps": n_steps,
                              "clip_frac_m3": 0.0, "kl_ref_m3": 0.0, "kl_m2": 0.0,
                              "reason": "no_informative_trajectories", "stopped_stale": False})
            break
        z = torch.zeros((), dtype=torch.float32, device=params[0].device)
        s3 = kr3 = ko = clip_count = entropy3 = z
        s2 = kr2 = z
        opt.zero_grad()
        if train_m3 and records:
            s3, kr3, ko, clip_count, entropy3 = _vectorized_m3_terms(
                jpol, records, weights, T, eps, clip_eps,
                float(JG.C.TO1_R14_BETA_M3))
        if train_m2 and records:
            s2, kr2, _n_m2 = _vectorized_m2_terms(
                jpol, records, weights, float(JG.C.TO1_R13_TEMP_M2), clip_eps,
                float(JG.C.TO1_R14_LAMBDA_M2), float(JG.C.TO1_R14_BETA_M2))
        loss = -s3 - float(JG.C.TO1_R14_LAMBDA_M2) * s2
        if train_m3 and records:
            loss = loss + float(JG.C.TO1_R14_BETA_M3) * kr3
        if train_m2:
            loss = loss + float(JG.C.TO1_R14_BETA_M2) * kr2
        def grad_norm(group):
            vals = [p.grad.detach().float().norm(2) ** 2 for p in group if p.grad is not None]
            return float(torch.sqrt(sum(vals)).detach()) if vals else 0.0
        gn2, gn3 = grad_norm(params_m2), grad_norm(params_m3)
        gn = float(nn.utils.clip_grad_norm_(params, 10.0)); opt.step()
        per_epoch.append({"epoch": ep, "loss": float(loss.detach()),
                          "n_informative_trajectories": count_inf2, "n_steps": n_steps,
                          "n_info_steps": count_inf2, "grad_norm": gn,
                          "kl_ref_m3": float(kr3.detach()), "kl_old": float(ko.detach()),
                          "kl_m2": float(kr2.detach()),
                          "entropy_m3": float(entropy3.detach()),
                          "clip_frac_m3": float(clip_count.detach()) / max(n_info_traj, 1),
                          "n_informative_trajectories_m2": count_inf2,
                          "grad_norm_m2": gn2, "grad_norm_m3": gn3,
                          "stopped_stale": False, "reason": ""})
    return {"n_informative_trajectories": n_info_traj, "n_trajectories": len(bundles),
            "n_steps": n_steps, "epochs": per_epoch,
            "n_informative_trajectories_m2": count_inf2,
            "optimizer_persistent": bool(optimizer_persistent),
            "optimizer_state_entries": len(opt.state),
            "optimizer_param_groups": [g.get("actor") for g in opt.param_groups],
            "final_kl_m3": per_epoch[-1]["kl_ref_m3"] if per_epoch else None,
            "final_kl_m2": per_epoch[-1]["kl_m2"] if per_epoch else None}
def structured_gpu_update_joint(jpol, groups, stage="C", T=None, eps=None,
                                clip_eps=None, seeds=None, epochs=None,
                                log_prefix="[t2d-gpu]", credit="stagewise",
                                optimizer_persistent=True, **_kwargs):
    """Signature-compatible GPU update preserving canonical stagewise reduction."""
    if credit != "stagewise":
        raise ValueError("T2-D supports stagewise A2/A3 credit only")
    dev = _device()
    start = time.time()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    gp = _gpu_policy(jpol, dev)
    T = float(T if T is not None else JG.C.TO1_R13_TEMP)
    eps = float(eps if eps is not None else JG.C.TO1_R13_MIX_EPS)
    clip_eps = float(clip_eps if clip_eps is not None else JG.C.TO1_R13_CLIP_EPS)
    epochs = int(epochs if epochs is not None else JG.C.TO1_R13_UPDATE_EPOCHS)
    samples, stop_sample = [], threading.Event()
    def sample_gpu():
        while not stop_sample.wait(0.1):
            value = _gpu_utilization_sample(dev)
            if value is not None:
                samples.append(value)
    sampler = None
    if dev.type == "cuda":
        sampler = threading.Thread(target=sample_gpu, daemon=True)
        sampler.start()
    batched = (hasattr(gp.m2, "final_logits_batched") and
               hasattr(gp.m3, "action_and_base_logits_batched"))
    try:
        if batched:
            out = _batched_update(
                gp, groups, stage, T, eps, clip_eps, seeds, epochs,
                optimizer_persistent)
        else:
            moved = _move_groups(groups, dev)
            out = _CANONICAL_UPDATE(
                gp, moved, stage=stage, T=T, eps=eps, clip_eps=clip_eps,
                seeds=seeds, epochs=epochs, log_prefix=log_prefix, credit=credit,
                optimizer_persistent=optimizer_persistent)
    finally:
        stop_sample.set()
        if sampler is not None:
            sampler.join(timeout=2.0)
    _copy_back(gp, jpol)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    measured_groups = groups if batched else moved
    n_prop = sum(int(r["F_pool"].shape[0]) for g in measured_groups for tr in g["trajs"]
                 for r in tr["steps"])
    n_roots = sum(int(r.get("m2_rec", {}).get("n_cand", 0))
                  for g in measured_groups for tr in g["trajs"] for r in tr["steps"])
    out["_monitor"] = {
        "device": str(dev), "structured_set_update": True,
        "bucketed_actor_forward": bool(batched),
        "vectorized_grpo_loss": bool(batched),
        "packed_proposals": n_prop, "packed_roots": n_roots,
        "transfer_and_update_s": time.time() - start,
        "optimizer_persistent": bool(optimizer_persistent),
        "gpu_peak_memory_mb": (float(torch.cuda.max_memory_allocated(dev)) / 1024.0 ** 2
                               if dev.type == "cuda" else 0.0),
        "gpu_utilization_pct_sample": _gpu_utilization_sample(dev),
        "gpu_utilization_update_mean_pct": (
            float(sum(samples) / len(samples)) if samples else None),
        "gpu_utilization_update_peak_pct": (max(samples) if samples else None),
        "gpu_utilization_update_samples": len(samples),
    }
    return out


__all__ = ["set_runtime", "set_microbatch_rows", "structured_gpu_update_joint",
           "optimizer_state_cpu"]
