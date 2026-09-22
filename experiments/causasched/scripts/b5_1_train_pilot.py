#!/usr/bin/env python3
"""B5.1 -- Causal Influence Attribution, Phase-1 minimal closed loop.

ONE shared attribution theta across all 14 TRAIN states (+ 3 VAL zero-shot).
Independent Route-II single data path (NOT the old root_utility shard):

  * supervision target  = appearance_delta  (VALID_ANCHORED rows only).
  * grouping            = (state_id, appearance_anchor_id) = one appearance A.
  * candidate -> node   : candidate_operation -> "operation:<op>" -> node index.
  * anchor  -> block    : appearance_anchor_id -> appearance_block_ids index.
  * score               = r_u(A) from frozen-c reverse propagation + learned e.
  * loss                = appearance-group tie-aware pairwise ranking ONLY.
                          priority_delta / makespan_delta NOT in the loss.
  * I_uv                = OFF.  pair_*.jsonl is never loaded.  Reasoner frozen.

Sign convention (B5-LABEL-CONTRACT-1.0):  appearance_delta = before - after,
positive = appearance WEAKENED (good).  A "top contributor" is the candidate
whose intervention most weakens A, i.e. the LARGEST appearance_delta.  Ranking
target U = appearance_delta.

Five gates (fixed reading order):
  Gate 1  single attribution learns   (TRAIN pair-acc up, VAL not random)
  Gate 2  same-theta VAL zero-shot     (no grad, same params)
  Gate 3  full beats no_appearance     (+ bit-level identity check)  <- R2 lesson
  Gate 4  physics ablation             (full vs no_physics)
  Gate 5  rule-wise breakdown          (A1/A2/A4 not dominated by one rule)

identified=false, formal_test_access=0.

Run:  PYTHONPATH=src .venv/bin/python scripts/b5_1_train_pilot.py [--max-steps N]
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEACHER_DIR = ROOT / "outputs" / "b5_route2_teacher"
OUT_DIR = ROOT / "outputs" / "b5_1_pilot"

LABEL_CONTRACT = "B5-LABEL-CONTRACT-1.1"  # A1 admission-gate bump (T1-B5-A1)
TIE_EPS = 1e-6  # |Δ_i - Δ_j| <= TIE_EPS -> no strict order imposed
FORMAL_TRAINING = 1
FORMAL_TEST_ACCESS = 0
IDENTIFIED = False


def _imports():
    import torch
    from torch.nn import functional as F

    from causal_schedule_lab.ir_adapters.fjsp_drl import load_fjsp_problem
    from causal_schedule_lab.solvers.dispatching import solve_dispatching
    from causal_schedule_lab.symptom_pruning import diagnose_and_prune
    from causal_schedule_lab.sg_sct_data_v1_3 import (
        compile_sg_sct_input_v1_3,
        to_sg_sct_batch_v1_3,
    )
    from causal_schedule_lab.sg_sct_model_v5 import build_m2_runtime_context
    from causal_schedule_lab.b5_attribution import compute_c_prior, from_manifest_b5
    from causal_schedule_lab.validation import schedule_hash

    return (
        torch, F, load_fjsp_problem, solve_dispatching, diagnose_and_prune,
        compile_sg_sct_input_v1_3, to_sg_sct_batch_v1_3, build_m2_runtime_context,
        compute_c_prior, from_manifest_b5, schedule_hash,
    )


(
    torch, F, load_fjsp_problem, solve_dispatching, diagnose_and_prune,
    compile_sg_sct_input_v1_3, to_sg_sct_batch_v1_3, build_m2_runtime_context,
    compute_c_prior, from_manifest_b5, schedule_hash,
) = _imports()


# ---------------------------------------------------------------------------
# instance path resolution (from the frozen manifest split)
# ---------------------------------------------------------------------------
def _instance_paths() -> dict:
    """Resolve every TRAIN/VAL instance_id to its .fjs path.

    Uses the AUTHORITATIVE (instance_id, fjs_path) tuples from the teacher
    generation script -- the exact list the frozen dataset was produced from.
    No heuristic filename matching (the id<->stem transform is not regular,
    e.g. Behnke_Behnke_m40_21 <-> Behnke_m40_Behnke21.fjs).
    """
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location(
        "_b5_teacher_gen", str(ROOT / "scripts" / "b5_route2_teacher_generation.py")
    )
    gen = _ilu.module_from_spec(spec)
    spec.loader.exec_module(gen)

    manifest = json.loads((TEACHER_DIR / "manifest.json").read_text())
    split = manifest["split_definition"]

    out = {}
    for sp, tuples in (("train", gen.TRAIN_INSTANCES), ("val", gen.VAL_INSTANCES)):
        for iid, rel in tuples:
            out[iid] = {"instance_id": iid, "split": sp, "path": str(ROOT / rel)}

    # cross-check against the frozen manifest split (fail-closed if drifted)
    for sp in ("train", "val"):
        want = set(split[sp])
        have = {i for i, v in out.items() if v["split"] == sp}
        if want != have:
            raise RuntimeError(
                f"split drift {sp}: manifest={sorted(want)} script={sorted(have)}"
            )
    return out, manifest


# ---------------------------------------------------------------------------
# state reconstruction (deterministic; ONE shared model rebound per state)
# ---------------------------------------------------------------------------
def build_state(inst: dict) -> dict:
    problem = load_fjsp_problem(inst["path"], problem_id=inst["instance_id"])
    schedule = solve_dispatching(problem, rule="earliest_finish")
    appearance = diagnose_and_prune(problem, schedule).model_dump(mode="json")
    bundle = compile_sg_sct_input_v1_3(
        problem, schedule, appearance, case_id=f"S::{inst['instance_id']}::b5"
    )
    node_ids = tuple(bundle.manifest["id_spaces"]["node_ids"])
    block_ids = tuple(bundle.manifest["id_spaces"]["appearance_block_ids"])
    context = build_m2_runtime_context(problem, schedule, appearance, block_ids=block_ids)
    batch = to_sg_sct_batch_v1_3(bundle, device="cpu")
    c_prior = compute_c_prior(problem, schedule, node_ids)
    return {
        "inst": inst, "bundle": bundle, "batch": batch, "context": context,
        "node_ids": node_ids, "block_ids": block_ids, "c_prior": c_prior,
        "node_index": {n: i for i, n in enumerate(node_ids)},
        "block_index": {b: i for i, b in enumerate(block_ids)},
        "schedule_hash": schedule_hash(schedule),
        # scope-layer needs (round 2, model untouched):
        "problem": problem, "schedule": schedule, "appearance": appearance,
    }


# ---------------------------------------------------------------------------
# teacher single-row loading -> per-group supervision (VALID_ANCHORED only)
# ---------------------------------------------------------------------------
def load_teacher_groups(split: str, states: dict) -> dict:
    """Return {state_id: [group, ...]}, each group a dict with:

      appearance_id, rule, block_index (in model),
      node_indices [k] (candidate operations mapped to graph nodes),
      targets [k]  (appearance_delta, VALID_ANCHORED only),
      n_rankable, n_pairs.
    """
    path = TEACHER_DIR / f"single_{split}.jsonl"
    rows = [json.loads(l) for l in path.open()]
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[(r["instance_id"], r["appearance_anchor_id"])].append(r)

    out = collections.defaultdict(list)
    dropped_node = 0
    dropped_block = 0
    for (iid, anchor), grp in by_group.items():
        # find the matching state (instance_id -> state key)
        st = None
        for sid, s in states.items():
            if s["inst"]["instance_id"] == iid:
                st = s
                break
        if st is None:
            continue
        if anchor not in st["block_index"]:
            dropped_block += 1
            continue
        b = st["block_index"][anchor]
        node_idx = []
        targets = []
        rule = grp[0]["appearance_rule"]
        for r in grp:
            if r["status"] != "VALID_ANCHORED":
                continue
            if r["appearance_delta"] is None:
                continue
            node_id = f"operation:{r['candidate_operation']}"
            ni = st["node_index"].get(node_id)
            if ni is None:
                dropped_node += 1
                continue
            node_idx.append(ni)
            targets.append(float(r["appearance_delta"]))
        if len(node_idx) < 2:
            continue  # not rankable
        # count valid pairs (strict order after tie_eps)
        n_pairs = 0
        for i in range(len(targets)):
            for j in range(len(targets)):
                if targets[i] - targets[j] > TIE_EPS:
                    n_pairs += 1
        if n_pairs == 0:
            continue
        out[iid].append({
            "appearance_id": anchor, "rule": rule, "block_index": b,
            "node_indices": torch.tensor(node_idx, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.float64),
            "n_rankable": len(node_idx), "n_pairs": n_pairs,
        })
    return out, {"dropped_node": dropped_node, "dropped_block": dropped_block}


# ---------------------------------------------------------------------------
# loss + metrics (appearance-group tie-aware pairwise ranking)
# ---------------------------------------------------------------------------
def group_ranking_loss(scores_block, group):
    """tie-aware pairwise logistic ranking within one appearance group.

    scores_block : [N] = r_u(A) for every node under this block.
    group        : has node_indices [k], targets [k].
      U_i > U_j (by > TIE_EPS)  ->  softplus(-(s_i - s_j))
      |U_i - U_j| <= TIE_EPS    ->  no constraint.
    """
    s = scores_block[group["node_indices"]].double()  # [k]
    U = group["targets"]  # [k]
    dU = U.unsqueeze(1) - U.unsqueeze(0)
    dS = s.unsqueeze(1) - s.unsqueeze(0)
    pos = dU > TIE_EPS
    if not pos.any():
        return s.sum() * 0.0
    return F.softplus(-dS[pos]).mean()


def group_metrics(scores_block, group):
    s = scores_block[group["node_indices"]].double()
    U = group["targets"]
    dU = U.unsqueeze(1) - U.unsqueeze(0)
    dS = s.unsqueeze(1) - s.unsqueeze(0)
    pos = dU > TIE_EPS
    m = {"rule": group["rule"], "n_rankable": group["n_rankable"],
         "n_pairs": group["n_pairs"], "pair_acc": None,
         "top1_correct": None, "spearman": None, "ndcg_at_3": None}
    if pos.any():
        m["pair_acc"] = float((dS[pos] > 0).double().mean().item())
    # top-1 contributor recall: is argmax(score) the argmax(U)?
    best_u = int(torch.argmax(U).item())
    top1_pred = int(torch.argmax(s).item())
    # allow ties in U at the top
    u_max = U.max()
    m["top1_correct"] = float(1.0 if U[top1_pred] >= u_max - TIE_EPS else 0.0)
    # spearman rho between s and U
    m["spearman"] = _spearman(s, U)
    m["ndcg_at_3"] = _ndcg(s, U, k=3)
    return m


def _spearman(a, b):
    import torch as _t
    n = a.numel()
    if n < 2:
        return None
    ra = _t.argsort(_t.argsort(a)).double()
    rb = _t.argsort(_t.argsort(b)).double()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm())
    if denom <= 0:
        return 0.0
    return float((ra @ rb / denom).item())


def _ndcg(s, U, k=3):
    import torch as _t
    n = s.numel()
    rel = U - U.min()  # non-negative gains
    order = _t.argsort(s, descending=True)
    kk = min(k, n)
    disc = _t.log2(_t.arange(2, kk + 2, dtype=_t.double))
    dcg = (rel[order][:kk] / disc).sum()
    ideal = _t.sort(rel, descending=True).values[:kk]
    idcg = (ideal / disc).sum()
    return float((dcg / idcg).item()) if idcg > 0 else 0.0


def aggregate(all_group_metrics):
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(sum(xs) / len(xs)) if xs else None
    by_rule = collections.defaultdict(list)
    for m in all_group_metrics:
        by_rule[m["rule"]].append(m)
    agg = {
        "n_groups": len(all_group_metrics),
        "n_pairs": sum(m["n_pairs"] for m in all_group_metrics),
        "mean_pair_acc": mean([m["pair_acc"] for m in all_group_metrics]),
        "mean_spearman": mean([m["spearman"] for m in all_group_metrics]),
        "mean_top1_recall": mean([m["top1_correct"] for m in all_group_metrics]),
        "mean_ndcg_at_3": mean([m["ndcg_at_3"] for m in all_group_metrics]),
        "by_rule": {},
    }
    for rule, ms in sorted(by_rule.items()):
        agg["by_rule"][rule] = {
            "n_groups": len(ms),
            "n_pairs": sum(m["n_pairs"] for m in ms),
            "mean_pair_acc": mean([m["pair_acc"] for m in ms]),
            "mean_spearman": mean([m["spearman"] for m in ms]),
            "mean_top1_recall": mean([m["top1_correct"] for m in ms]),
        }
    return agg


# ---------------------------------------------------------------------------
# forward: bind + attribution
# ---------------------------------------------------------------------------
def forward_scores(model, st, *, grad: bool):
    model.bind_runtime_context(st["context"], node_ids=list(st["node_ids"]))
    if grad:
        return model.attribution_forward(st["batch"], st["c_prior"]).per_block_candidate_score
    with torch.no_grad():
        return model.attribution_forward(st["batch"], st["c_prior"]).per_block_candidate_score


def state_loss(model, st, groups):
    scores = forward_scores(model, st, grad=True)
    losses = [group_ranking_loss(scores[g["block_index"]], g) for g in groups]
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def eval_state(model, st, groups):
    scores = forward_scores(model, st, grad=False)
    return [group_metrics(scores[g["block_index"]], g) for g in groups]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    insts, manifest = _instance_paths()
    train_ids = [i for i in insts.values() if i["split"] == "train"]
    val_ids = [i for i in insts.values() if i["split"] == "val"]
    assert len(train_ids) == 14 and len(val_ids) == 3, (len(train_ids), len(val_ids))

    print("[build] reconstructing states ...", flush=True)
    states = {}
    for inst in train_ids + val_ids:
        t0 = time.time()
        states[inst["instance_id"]] = build_state(inst)
        print(f"  {inst['instance_id']:26s} {round(time.time()-t0,1)}s", flush=True)

    train_groups, tinfo = load_teacher_groups("train", states)
    val_groups, vinfo = load_teacher_groups("val", states)
    n_train_groups = sum(len(v) for v in train_groups.values())
    n_val_groups = sum(len(v) for v in val_groups.values())
    n_train_pairs = sum(g["n_pairs"] for v in train_groups.values() for g in v)
    n_val_pairs = sum(g["n_pairs"] for v in val_groups.values() for g in v)
    print(f"[data] TRAIN groups={n_train_groups} pairs={n_train_pairs} drop={tinfo}")
    print(f"[data] VAL   groups={n_val_groups} pairs={n_val_pairs} drop={vinfo}")

    # ONE shared model
    model = from_manifest_b5(states[train_ids[0]["instance_id"]]["bundle"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    def eval_all(m):
        m.eval()
        tr = []
        for iid, gs in train_groups.items():
            tr += eval_state(m, states[iid], gs)
        va = []
        for iid, gs in val_groups.items():
            va += eval_state(m, states[iid], gs)
        return aggregate(tr), aggregate(va), tr, va

    # step-0 baseline
    tr0, va0, _, _ = eval_all(model)
    print(f"[step 0] TRAIN pair_acc={tr0['mean_pair_acc']} spearman={tr0['mean_spearman']} "
          f"| VAL pair_acc={va0['mean_pair_acc']} spearman={va0['mean_spearman']}")

    log = []
    first_loss = last_loss = None
    for step in range(1, args.max_steps + 1):
        model.train()
        opt.zero_grad()
        sls = []
        for iid in train_groups:
            sls.append(state_loss(model, states[iid], train_groups[iid]))
        loss = torch.stack(sls).mean()
        if not torch.isfinite(loss):
            print(f"[step {step}] NON-FINITE loss -> stop"); break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.item())
        if first_loss is None:
            first_loss = last_loss
        if step % args.eval_every == 0 or step == args.max_steps:
            tr, va, _, _ = eval_all(model)
            print(f"[step {step}] loss={round(last_loss,4)} "
                  f"TRAIN pa={_r(tr['mean_pair_acc'])} sp={_r(tr['mean_spearman'])} "
                  f"top1={_r(tr['mean_top1_recall'])} "
                  f"| VAL pa={_r(va['mean_pair_acc'])} sp={_r(va['mean_spearman'])} "
                  f"top1={_r(va['mean_top1_recall'])}", flush=True)
            log.append({"step": step, "loss": last_loss, "train": tr, "val": va})

    # ---- final eval + gates ----
    tr_f, va_f, tr_groups_m, va_groups_m = eval_all(model)

    gates = run_gates(model, states, train_groups, val_groups, tr0, tr_f, va_f)

    result = {
        "task": "B5.1 causal influence attribution -- Phase 1 minimal closed loop",
        "label_contract": LABEL_CONTRACT,
        "seed": args.seed, "max_steps": args.max_steps, "lr": args.lr,
        "first_loss": first_loss, "last_loss": last_loss,
        "data": {"n_train_groups": n_train_groups, "n_val_groups": n_val_groups,
                 "n_train_pairs": n_train_pairs, "n_val_pairs": n_val_pairs,
                 "train_drop": tinfo, "val_drop": vinfo},
        "step0": {"train": tr0, "val": va0},
        "final": {"train": tr_f, "val": va_f},
        "gates": gates,
        "c_frozen": True, "I_uv": "OFF", "reasoner": "FROZEN",
        "formal_training": FORMAL_TRAINING, "formal_test_access": FORMAL_TEST_ACCESS,
        "identified": IDENTIFIED,
    }
    (OUT_DIR / "log.jsonl").write_text("\n".join(json.dumps(x) for x in log) + "\n")
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    torch.save(model.state_dict(), OUT_DIR / "b5_1_shared.pt")

    print("\n==================== GATES ====================")
    for k, v in gates.items():
        print(f"  {k}: {v['verdict']}  {v.get('detail','')}")
    print("Saved:", OUT_DIR / "result.json")


def _r(x):
    return round(x, 3) if x is not None else None


def run_gates(model, states, train_groups, val_groups, tr0, tr_f, va_f):
    gates = {}

    # Gate 1: single attribution learns (TRAIN pair-acc up from step0; not random)
    up = (tr_f["mean_pair_acc"] or 0) - (tr0["mean_pair_acc"] or 0)
    g1 = (tr_f["mean_pair_acc"] or 0) > 0.55 and up > 0.02
    gates["gate1_single_attribution_learns"] = {
        "verdict": "PASS" if g1 else "FAIL",
        "detail": f"TRAIN pair_acc {_r(tr0['mean_pair_acc'])}->{_r(tr_f['mean_pair_acc'])} (Δ{_r(up)})",
    }

    # Gate 2: same-theta VAL zero-shot better than random
    g2 = (va_f["mean_pair_acc"] or 0) > 0.55
    gates["gate2_val_zero_shot"] = {
        "verdict": "PASS" if g2 else "FAIL",
        "detail": f"VAL pair_acc={_r(va_f['mean_pair_acc'])} spearman={_r(va_f['mean_spearman'])}",
    }

    # Gate 3: full beats no_appearance + bit-level identity check.
    model.eval()
    def eval_ablation(use_app, use_phys):
        model.set_b5_ablation(use_appearance=use_app, use_physics=use_phys)
        allm = []
        raw = {}
        for iid, gs in train_groups.items():
            st = states[iid]
            model.bind_runtime_context(st["context"], node_ids=list(st["node_ids"]))
            with torch.no_grad():
                sc = model.attribution_forward(st["batch"], st["c_prior"]).per_block_candidate_score
            raw[iid] = sc.clone()
            for g in gs:
                allm.append(group_metrics(sc[g["block_index"]], g))
        model.set_b5_ablation(use_appearance=True, use_physics=True)
        return aggregate(allm), raw

    full_agg, full_raw = eval_ablation(True, True)
    noapp_agg, noapp_raw = eval_ablation(False, True)
    # bit-level identity: are full and no_appearance scores identical anywhere?
    identical = all(
        torch.allclose(full_raw[iid], noapp_raw[iid]) for iid in full_raw
    )
    g3 = ((full_agg["mean_pair_acc"] or 0) > (noapp_agg["mean_pair_acc"] or 0) + 0.02) and not identical
    gates["gate3_full_beats_no_appearance"] = {
        "verdict": "PASS" if g3 else "FAIL",
        "detail": f"full pair_acc={_r(full_agg['mean_pair_acc'])} vs "
                  f"no_appearance={_r(noapp_agg['mean_pair_acc'])} | bit_identical={identical}",
    }

    # Gate 4: physics ablation (full vs no_physics)
    nophys_agg, nophys_raw = eval_ablation(True, False)
    phys_identical = all(
        torch.allclose(full_raw[iid], nophys_raw[iid]) for iid in full_raw
    )
    g4 = ((full_agg["mean_pair_acc"] or 0) >= (nophys_agg["mean_pair_acc"] or 0)) and not phys_identical
    gates["gate4_physics_contributes"] = {
        "verdict": "PASS" if g4 else "FAIL",
        "detail": f"full pair_acc={_r(full_agg['mean_pair_acc'])} vs "
                  f"no_physics={_r(nophys_agg['mean_pair_acc'])} | bit_identical={phys_identical}",
    }

    # Gate 5: rule-wise breakdown not dominated by one rule
    rules = tr_f["by_rule"]
    accs = {r: rules[r]["mean_pair_acc"] for r in rules if rules[r]["mean_pair_acc"] is not None}
    learned = {r: (a or 0) > 0.55 for r, a in accs.items()}
    g5 = sum(learned.values()) >= 2  # at least 2 of A1/A2/A4 learn
    gates["gate5_rule_wise_not_dominated"] = {
        "verdict": "PASS" if g5 else "FAIL",
        "detail": {r: {"n_groups": rules[r]["n_groups"], "pair_acc": _r(rules[r]["mean_pair_acc"])}
                   for r in sorted(rules)},
    }
    return gates


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())
