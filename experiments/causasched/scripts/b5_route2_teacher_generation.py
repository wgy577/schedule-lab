#!/usr/bin/env python
"""B5-ROUTE-II-TEACHER -- round 2: formal teacher dataset production.

Executes the frozen B5-LABEL-CONTRACT-1.0 (docs/T1_MODEL_B5_LABEL_CONTRACT_FREEZE.md):
- single rows: full generation per frozen schema, ALL retained blocks per state;
- pair rows: deterministic STRUCTURAL sampling only (PAIR_SAMPLING_V1) --
  graph-related pairs + negative/control pairs; no CE ranking, no M2 Top-K;
- joint deltas: real execute_atoms replay; I_uv = joint - d_u - d_v;
- split: 14 TRAIN + 3 VAL (T1-DATA prereg, verbatim); Formal TEST sealed (0 access);
- after production: the 9 contract gates, nothing more;
- dataset fingerprint/manifest locks contract versions + row counts + file hash.

Resumable: each state writes an atomic per-state checkpoint; reruns skip done states.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from b5_label_feasibility_audit import (  # noqa: E402
    _edit_to_atom,
    _imports,
    block_candidates,
    block_ops,
    block_strength,
    match_block_after,
    machine_of,
)

from causal_schedule_lab.teacher.anchored_appearance_evaluator import (  # noqa: E402
    anchored_appearance_value,
)

OUT_DIR = ROOT / "outputs" / "b5_route2_teacher"
CKPT_DIR = OUT_DIR / "states"

# ---------------------------------------------------------------------------
# Frozen configuration (manifest-locked)
# ---------------------------------------------------------------------------
# T1-B5-A1-MAKESPAN-RELEVANT-GAP (2026-08-26): A1 detector gained an admission
# gate -- a gap is an active A1 block only when the "right" op (after the gap)
# has deterministic MakespanRelevance M_j >= tau_m.  Label philosophy unchanged
# (appearance_delta = before - after; makespan_delta independent); the version
# bump only records that the detector contract changed and the dataset was
# regenerated under it.
LABEL_CONTRACT = "B5-LABEL-CONTRACT-1.1"
A1_RULE_VERSION = "A1-RELEVANT-GAP-1.1"
PAIR_SAMPLING_VERSION = "PAIR_SAMPLING_V1"
CANDIDATE_ENUM_VERSION = "CAND_ENUM_V1"
SCHEDULE_RULE = "earliest_finish"
MAX_CANDIDATES = 15
MAX_GRAPH_PAIRS_PER_BLOCK = 6
MAX_NEGATIVE_PAIRS_PER_BLOCK = 2
INTERACTION_EPS = 1e-6
DYNAMIC_JACCARD_THRESHOLD = 0.5  # secondary audit only

# T1-DATA prereg split, verbatim (14 TRAIN + 3 VAL; TEST sealed).
TRAIN_INSTANCES: tuple[tuple[str, str], ...] = (
    ("Behnke_Behnke_m40_21", "instances/behnke_gehring/Behnke_m40_Behnke21.fjs"),
    ("Behnke_Behnke_m60_41", "instances/behnke_gehring/Behnke_m60_Behnke41.fjs"),
    ("Brandimarte_Mk1", "instances/brandimarte/BrandimarteMk1.fjs"),
    ("Brandimarte_Mk3", "instances/brandimarte/BrandimarteMk3.fjs"),
    ("Brandimarte_Mk9", "instances/brandimarte/BrandimarteMk9.fjs"),
    ("DPdata_DPpaulli1a", "instances/dpplaulli/DPpaulli1a.fjs"),
    ("DPdata_DPpaulli10a", "instances/dpplaulli/DPpaulli10a.fjs"),
    ("Fattahi_Fattahi11", "instances/fattahi/Fattahi11.fjs"),
    ("Fattahi_Fattahi15", "instances/fattahi/Fattahi15.fjs"),
    ("Hurink_Edata1", "instances/hurink_edata/HurinkEdata1.fjs"),
    ("Hurink_Edata10", "instances/hurink_edata/HurinkEdata10.fjs"),
    ("Hurink_Rdata1", "instances/hurink_rdata/HurinkRdata1.fjs"),
    ("Hurink_Vdata1", "instances/hurink_vdata/HurinkVdata1.fjs"),
    ("Hurink_Vdata10", "instances/hurink_vdata/HurinkVdata10.fjs"),
)
VAL_INSTANCES: tuple[tuple[str, str], ...] = (
    ("Behnke_Behnke1", "instances/behnke_gehring/Behnke1.fjs"),
    ("Brandimarte_Mk11", "instances/brandimarte/BrandimarteMk11.fjs"),
    ("Hurink_Rdata10", "instances/hurink_rdata/HurinkRdata10.fjs"),
)


# ---------------------------------------------------------------------------
# Deterministic structural pair selection (no CE values, no model outputs)
# ---------------------------------------------------------------------------
def _graph_related(problem, amap, u: str, v: str) -> bool:
    opmap = problem.operation_map()
    same_job = opmap[u].job_id == opmap[v].job_id
    mu = machine_of(problem, amap[u])
    mv = machine_of(problem, amap[v])
    same_machine = mu is not None and mu == mv
    hop = (v in opmap[u].predecessors) or (u in opmap[v].predecessors)
    return bool(same_job or same_machine or hop)


def _select_pairs(problem, amap, cand_order: list[str],
                  feasible: set[str]) -> list[tuple[str, str, str]]:
    """Deterministic structural pairs among feasible candidates, in cand order.

    Returns (u, v, kind) with kind in {"graph_related", "negative_control"}.
    Selection depends only on graph structure + candidate order -- never on CE
    deltas or any model output.
    """
    pairs: list[tuple[str, str, str]] = []
    neg: list[tuple[str, str, str]] = []
    for i in range(len(cand_order)):
        for j in range(i + 1, len(cand_order)):
            u, v = cand_order[i], cand_order[j]
            if u not in feasible or v not in feasible:
                continue
            if _graph_related(problem, amap, u, v):
                if sum(1 for _, _, k in pairs if k == "graph_related") < MAX_GRAPH_PAIRS_PER_BLOCK:
                    pairs.append((u, v, "graph_related"))
            else:
                if len(neg) < MAX_NEGATIVE_PAIRS_PER_BLOCK:
                    neg.append((u, v, "negative_control"))
    return pairs + neg


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _sign(x: float | None) -> str:
    if x is None:
        return "missing"
    if abs(x) < 1e-9:
        return "zero"
    return "positive" if x > 0 else "negative"


def _run_state(
    split: str, instance_id: str, fjs_path: str, ctx: dict[str, Any]
) -> tuple[list[dict], list[dict]]:
    (
        load_fjsp_problem, solve_dispatching, diagnose_and_prune,
        LegalEditEnumerator, AtomicCounterfactualExecutor, DecisionAtom,
        schedule_hash,
    ) = ctx["imports"]
    executor: AtomicCounterfactualExecutor = ctx["executor"]

    problem = load_fjsp_problem(fjs_path, problem_id=instance_id)
    schedule = solve_dispatching(problem, rule=SCHEDULE_RULE)
    base_ms = int(schedule.makespan)
    state_id = f"S::{instance_id}::{schedule_hash(schedule)[:16]}"
    amap = schedule.assignment_map()

    snap = diagnose_and_prune(problem, schedule)
    blocks = sorted(snap.retained, key=lambda b: b.block.block_id)

    enum = LegalEditEnumerator(problem, schedule)
    edits = enum.enumerate(
        tuple(sorted(amap)), request_route=True, request_seq_swap=True,
        request_seq_insert=True, request_timing_shift=False,
    )
    by_op: dict[str, list] = defaultdict(list)
    for e in edits:
        by_op[e.operation_id].append(e)

    single_rows: list[dict] = []
    pair_rows: list[dict] = []

    for block in blocks:
        bid = block.block.block_id
        rule = block.block.appearance_rules[0]
        before_val = float(block.block.appearance_values.get(rule, 0.0))
        before_pri = block_strength(block)[0]
        orig_ops = block_ops(block)

        cand_order = block_candidates(problem, schedule, block,
                                      max_cands=MAX_CANDIDATES)
        # ---- singles -------------------------------------------------
        atom_of: dict[str, Any] = {}
        kind_of: dict[str, str] = {}
        delta_of: dict[str, float | None] = {}
        feasible: set[str] = set()
        for op in cand_order:
            atom, kind = None, None
            for e in by_op.get(op, []):
                if e.edit_type == "ROUTE":
                    atom, kind = _edit_to_atom(e, DecisionAtom), "routing"
                    if atom is not None:
                        break
            if atom is None:
                for e in by_op.get(op, []):
                    atom = _edit_to_atom(e, DecisionAtom)
                    if atom is not None:
                        kind = ("routing" if e.edit_type == "ROUTE" else "sequencing")
                        break
            if atom is None:
                single_rows.append({
                    "state_id": state_id, "split": split, "instance_id": instance_id,
                    "appearance_anchor_id": bid, "appearance_rule": rule,
                    "original_members": sorted(orig_ops),
                    "candidate_operation": op, "decision_site": None,
                    "atomic_edit_id": None, "edit_kind": None,
                    "status": "NO_LEGAL_ATOM",
                })
                continue

            try:
                res = executor.execute_atom(problem, schedule, atom)
            except Exception as exc:
                single_rows.append({
                    "state_id": state_id, "split": split, "instance_id": instance_id,
                    "appearance_anchor_id": bid, "appearance_rule": rule,
                    "original_members": sorted(orig_ops),
                    "candidate_operation": op,
                    "decision_site": {"atom_id": atom.atom_id, "atom_type": atom.atom_type,
                                      "machine": atom.machine,
                                      "partner": atom.partner_operation},
                    "atomic_edit_id": atom.atom_id, "edit_kind": kind,
                    "status": "EXECUTION_FAILED", "error": type(exc).__name__,
                })
                continue
            if not (res.feasible and res.schedule is not None):
                single_rows.append({
                    "state_id": state_id, "split": split, "instance_id": instance_id,
                    "appearance_anchor_id": bid, "appearance_rule": rule,
                    "original_members": sorted(orig_ops),
                    "candidate_operation": op,
                    "decision_site": {"atom_id": atom.atom_id, "atom_type": atom.atom_type,
                                      "machine": atom.machine,
                                      "partner": atom.partner_operation},
                    "atomic_edit_id": atom.atom_id, "edit_kind": kind,
                    "status": "HARD_INFEASIBLE",
                })
                continue

            after = res.schedule
            after_ms = int(after.makespan)
            ev = anchored_appearance_value(problem, after, block)
            try:
                after_snap = diagnose_and_prune(problem, after)
                matched, jac = match_block_after(after_snap, orig_ops)
            except Exception:
                matched, jac = None, None
            dynamic_disappeared = matched is None or (
                jac is not None and jac < DYNAMIC_JACCARD_THRESHOLD)
            dynamic_pri = block_strength(matched)[0] if matched is not None else None

            rec = {
                "state_id": state_id, "split": split, "instance_id": instance_id,
                "appearance_anchor_id": bid, "appearance_rule": rule,
                "original_members": sorted(orig_ops),
                "candidate_operation": op,
                "decision_site": {"atom_id": atom.atom_id, "atom_type": atom.atom_type,
                                  "machine": atom.machine,
                                  "partner": atom.partner_operation},
                "atomic_edit_id": atom.atom_id, "edit_kind": kind,
                "before_anchored_value": before_val,
                "after_anchored_value": ev.value if ev.evaluable else None,
                "appearance_delta": (round(before_val - ev.value, 9)
                                     if ev.evaluable else None),
                "priority_delta": (round(before_pri - dynamic_pri, 9)
                                   if dynamic_pri is not None else None),
                "makespan_delta": base_ms - after_ms,
                "status": "VALID_ANCHORED" if ev.evaluable else "ANCHOR_UNEVALUABLE",
                "anchored_reason": ev.reason,
                "rematch_jaccard": round(jac, 4) if jac is not None else None,
                "dynamic_disappeared": bool(dynamic_disappeared),
                "schedule_source": f"deterministic-dispatching:{SCHEDULE_RULE}",
            }
            single_rows.append(rec)
            atom_of[op] = atom
            kind_of[op] = kind
            if ev.evaluable:
                delta_of[op] = round(before_val - ev.value, 9)
                feasible.add(op)
            else:
                delta_of[op] = None

        # ---- pairs (structural sampling; joint = real execute_atoms) --
        for u, v, pair_kind in _select_pairs(problem, amap, cand_order, feasible):
            au, av = atom_of[u], atom_of[v]
            try:
                res = executor.execute_atoms(problem, schedule, (au, av))
            except Exception as exc:
                pair_rows.append({
                    "state_id": state_id, "split": split, "instance_id": instance_id,
                    "appearance_anchor_id": bid, "appearance_rule": rule,
                    "u": u, "v": v, "atom_u": au.atom_id, "atom_v": av.atom_id,
                    "pair_kind": pair_kind, "status": "EXECUTION_FAILED",
                    "error": type(exc).__name__,
                })
                continue
            if not (res.feasible and res.schedule is not None):
                pair_rows.append({
                    "state_id": state_id, "split": split, "instance_id": instance_id,
                    "appearance_anchor_id": bid, "appearance_rule": rule,
                    "u": u, "v": v, "atom_u": au.atom_id, "atom_v": av.atom_id,
                    "pair_kind": pair_kind, "status": "HARD_INFEASIBLE",
                })
                continue
            after = res.schedule
            after_ms = int(after.makespan)
            ev = anchored_appearance_value(problem, after, block)
            try:
                after_snap = diagnose_and_prune(problem, after)
                matched, jac = match_block_after(after_snap, orig_ops)
            except Exception:
                matched, jac = None, None
            joint_delta = (round(before_val - ev.value, 9)
                           if ev.evaluable else None)
            du, dv = delta_of.get(u), delta_of.get(v)
            i_uv = (round(joint_delta - du - dv, 9)
                    if (joint_delta is not None and du is not None and dv is not None)
                    else None)
            interaction = None
            if i_uv is not None:
                if i_uv > INTERACTION_EPS:
                    interaction = "synergy"
                elif i_uv < -INTERACTION_EPS:
                    interaction = "interference"
                else:
                    interaction = "additive"
            pair_rows.append({
                "state_id": state_id, "split": split, "instance_id": instance_id,
                "appearance_anchor_id": bid, "appearance_rule": rule,
                "original_members": sorted(orig_ops),
                "u": u, "v": v,
                "atom_u": au.atom_id, "atom_v": av.atom_id,
                "edit_kind_u": kind_of[u], "edit_kind_v": kind_of[v],
                "pair_kind": pair_kind,
                "before_anchored_value": before_val,
                "joint_after_anchored_value": ev.value if ev.evaluable else None,
                "joint_appearance_delta": joint_delta,
                "joint_makespan_delta": base_ms - after_ms,
                "delta_u": du, "delta_v": dv,
                "I_uv": i_uv,
                "interaction_type": interaction,
                "status": "VALID_ANCHORED" if ev.evaluable else "ANCHOR_UNEVALUABLE",
                "anchored_reason": ev.reason,
                "rematch_jaccard": round(jac, 4) if jac is not None else None,
                "dynamic_disappeared": bool(
                    matched is None or (jac is not None and jac < DYNAMIC_JACCARD_THRESHOLD)),
            })

    return single_rows, pair_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = {"imports": _imports(), "executor": _imports()[4]()}

    jobs = [("train", iid, path) for iid, path in TRAIN_INSTANCES] + \
           [("val", iid, path) for iid, path in VAL_INSTANCES]

    t0 = time.time()
    for split, instance_id, fjs_path in jobs:
        ckpt = CKPT_DIR / f"{split}__{instance_id}.json"
        if ckpt.exists():
            print(f"[skip] {split}/{instance_id} (checkpoint exists)", flush=True)
            continue
        t_state = time.time()
        singles, pairs = _run_state(split, instance_id, fjs_path, ctx)
        payload = {
            "split": split, "instance_id": instance_id,
            "n_single": len(singles), "n_pair": len(pairs),
            "single": singles, "pair": pairs,
        }
        tmp = ckpt.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True))
        tmp.replace(ckpt)
        print(f"[done] {split}/{instance_id}: {len(singles)} singles, "
              f"{len(pairs)} pairs in {time.time() - t_state:.1f}s "
              f"(elapsed {time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------
    # merge + manifest + gates
    # ------------------------------------------------------------------
    all_single: list[dict] = []
    all_pair: list[dict] = []
    for split, instance_id, _ in jobs:
        ckpt = CKPT_DIR / f"{split}__{instance_id}.json"
        payload = json.loads(ckpt.read_text())
        all_single.extend(payload["single"])
        all_pair.extend(payload["pair"])

    for name, rows in (("single_train", [r for r in all_single if r["split"] == "train"]),
                       ("single_val", [r for r in all_single if r["split"] == "val"]),
                       ("pair_train", [r for r in all_pair if r["split"] == "train"]),
                       ("pair_val", [r for r in all_pair if r["split"] == "val"])):
        (OUT_DIR / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")

    manifest = {
        "label_contract": LABEL_CONTRACT,
        "split_definition": {
            "train": [iid for iid, _ in TRAIN_INSTANCES],
            "val": [iid for iid, _ in VAL_INSTANCES],
            "test": "SEALED (0 access)",
        },
        "state_source": f"deterministic-dispatching:{SCHEDULE_RULE}",
        "candidate_enumeration_version": (
            f"{CANDIDATE_ENUM_VERSION}: members+1-hop-ancestors, max {MAX_CANDIDATES}; "
            "LegalEditEnumerator route+seq_swap+seq_insert, timing_shift=off"),
        "pair_sampling_version": (
            f"{PAIR_SAMPLING_VERSION}: structural only (graph_related<="
            f"{MAX_GRAPH_PAIRS_PER_BLOCK}/block + negative_control<="
            f"{MAX_NEGATIVE_PAIRS_PER_BLOCK}/block); no CE ranking, no M2 Top-K"),
        "anchored_evaluator_version": "anchored_appearance_evaluator.py "
                                      + _sha256_file(ROOT / "src" / "causal_schedule_lab"
                                                     / "teacher" / "anchored_appearance_evaluator.py")[:16],
        "a1_rule_version": A1_RULE_VERSION,
        "a1_detector_sha": _sha256_file(ROOT / "src" / "causal_schedule_lab"
                                        / "symptom_pruning.py")[:16],
        "teacher_script_version": _sha256_file(Path(__file__))[:16],
        "row_counts": {
            "single_train": sum(1 for r in all_single if r["split"] == "train"),
            "single_val": sum(1 for r in all_single if r["split"] == "val"),
            "pair_train": sum(1 for r in all_pair if r["split"] == "train"),
            "pair_val": sum(1 for r in all_pair if r["split"] == "val"),
        },
    }
    for name in ("single_train", "single_val", "pair_train", "pair_val"):
        manifest[f"{name}_sha256"] = _sha256_file(OUT_DIR / f"{name}.jsonl")
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    # ---------------- 9 contract gates (nothing more) ------------------
    valid_single = [r for r in all_single if r["status"] == "VALID_ANCHORED"]
    valid_pair = [r for r in all_pair if r["status"] == "VALID_ANCHORED"]

    def _dist(rows, key):
        return dict(Counter(_sign(r.get(key)) for r in rows))

    rule_counts = Counter(r["appearance_rule"] for r in valid_single)
    per_rule_status = defaultdict(Counter)
    for r in all_single:
        per_rule_status[r["appearance_rule"]][r["status"]] += 1
    edit_kinds = Counter(r.get("edit_kind") for r in valid_single)
    interaction_dist = Counter(r["interaction_type"] for r in valid_pair
                               if r.get("interaction_type"))
    cand_mult = defaultdict(list)
    for r in all_single:
        cand_mult[(r["instance_id"], r["appearance_anchor_id"])].append(r["status"])
    multiplicity = {
        "n_blocks": len(cand_mult),
        "mean_candidates_per_block": round(
            sum(len(v) for v in cand_mult.values()) / max(len(cand_mult), 1), 3),
        "mean_valid_per_block": round(
            sum(1 for v in cand_mult.values() for s in v if s == "VALID_ANCHORED")
            / max(len(cand_mult), 1), 3),
    }

    def _split_stats(rows, key):
        out = {}
        for sp in ("train", "val"):
            sub = [r for r in rows if r["split"] == sp]
            out[sp] = {"n": len(sub), "sign": _dist(sub, key)}
        return out

    gates = {
        "1_anchored_label_coverage": {
            "valid": len(valid_single), "total": len(all_single),
            "frac": round(len(valid_single) / max(len(all_single), 1), 4),
            "status_dist": dict(Counter(r["status"] for r in all_single)),
        },
        "2_rule_coverage": {
            "valid_per_rule": dict(rule_counts),
            "per_rule_status": {k: dict(v) for k, v in per_rule_status.items()},
        },
        "3_sign_distribution": {
            "appearance_delta": _dist(valid_single, "appearance_delta"),
            "makespan_delta": _dist(valid_single, "makespan_delta"),
            "by_split": _split_stats(valid_single, "appearance_delta"),
        },
        "4_candidate_multiplicity": multiplicity,
        "5_routing_sequence_coverage": dict(edit_kinds),
        "6_pair_interaction_coverage": {
            "n_pairs_total": len(all_pair),
            "n_pairs_valid": len(valid_pair),
            "n_with_I_uv": sum(1 for r in valid_pair if r.get("I_uv") is not None),
            "interaction_dist": dict(interaction_dist),
            "frac_non_additive": round(
                sum(c for k, c in interaction_dist.items() if k != "additive")
                / max(sum(interaction_dist.values()), 1), 4),
            "pair_kind_dist": dict(Counter(r.get("pair_kind") for r in all_pair)),
            "pair_status_dist": dict(Counter(r["status"] for r in all_pair)),
        },
        "7_train_val_skew": {
            "valid_frac_by_split": {
                sp: round(sum(1 for r in all_single if r["split"] == sp
                              and r["status"] == "VALID_ANCHORED")
                          / max(sum(1 for r in all_single if r["split"] == sp), 1), 4)
                for sp in ("train", "val")
            },
            "rule_mix_train": dict(Counter(r["appearance_rule"] for r in valid_single
                                           if r["split"] == "train")),
            "rule_mix_val": dict(Counter(r["appearance_rule"] for r in valid_single
                                         if r["split"] == "val")),
        },
        "8_anchor_unevaluable": {
            "n": sum(1 for r in all_single if r["status"] == "ANCHOR_UNEVALUABLE"),
            "frac": round(sum(1 for r in all_single if r["status"] == "ANCHOR_UNEVALUABLE")
                          / max(len(all_single), 1), 4),
            "reasons": dict(Counter(r.get("anchored_reason") for r in all_single
                                    if r["status"] == "ANCHOR_UNEVALUABLE")),
        },
        "9_dynamic_rematch_audit": {
            "n_dynamic_disappeared": sum(1 for r in all_single
                                         if r.get("dynamic_disappeared")),
            "frac_dynamic_disappeared": round(
                sum(1 for r in all_single if r.get("dynamic_disappeared"))
                / max(len(all_single), 1), 4),
            "n_recovered_valid_when_gone": sum(
                1 for r in all_single
                if r.get("dynamic_disappeared") and r["status"] == "VALID_ANCHORED"),
        },
    }
    (OUT_DIR / "gates.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": manifest["row_counts"], "gates": gates},
                     indent=2, sort_keys=True))
    print(f"wall_seconds={time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
