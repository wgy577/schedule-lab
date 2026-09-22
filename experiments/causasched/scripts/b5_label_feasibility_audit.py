#!/usr/bin/env python
"""B5-LABEL-FEASIBILITY AUDIT -- 只验证 causal-influence-attribution 标签是否可生成。

目标（T1-MODEL-B5 §下一步，GPT 定调）：
    先证明"老师能不能存在"——有没有办法从现有 Frozen Local / atomic CE 产生可信的
    appearance-conditioned 归因监督。**不写任何模型代码，不训练。**

Probe 规模（GPT 定）：
    10 TRAIN states × 3-5 appearance blocks × top-15 candidate nodes ≈ 600 atomic eval。

禁止（hard）：
    改 M2 / Explorer / Reasoner；训练；backward/optimizer；TEST 访问。

回答四问（GPT 优先级：先 Q1 node-attribution label 存在性，edge transmission 后置）：
    Q1  appearance-conditioned CE 是否稳定？（动 j -> 表象 A 强度 before/after）
    Q3  node self-effect proxy coverage（等待/slack 覆盖 positive-attribution 节点多少）
    Q4  removal counterfactual feasibility（干预根因 -> 表象 A 是否改善）
    Q2  edge transmission decomposition（后置，本轮只报边传导的构造可行性）

结论三选一：
    A  ROUTE-II LABEL FEASIBLE
    B  ROUTE-I ONLY
    C  重新设计 attribution target
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "outputs" / "b5_label_feasibility_audit"

# Probe 规模（GPT 定，不过度纠结）
N_STATES = 10
N_BLOCKS_PER_STATE = 3
N_CANDIDATES_PER_BLOCK = 15

# 14 TRAIN 实例里选 10 个；避开 DPpaulli10a（stage_b 有 exp-overflow shim）。
PROBE_INSTANCES: tuple[tuple[str, str], ...] = (
    ("Behnke_Behnke_m40_21", "instances/behnke_gehring/Behnke_m40_Behnke21.fjs"),
    ("Behnke_Behnke_m60_41", "instances/behnke_gehring/Behnke_m60_Behnke41.fjs"),
    ("Brandimarte_Mk1", "instances/brandimarte/BrandimarteMk1.fjs"),
    ("Brandimarte_Mk3", "instances/brandimarte/BrandimarteMk3.fjs"),
    ("Brandimarte_Mk9", "instances/brandimarte/BrandimarteMk9.fjs"),
    ("DPdata_DPpaulli1a", "instances/dpplaulli/DPpaulli1a.fjs"),
    ("Fattahi_Fattahi11", "instances/fattahi/Fattahi11.fjs"),
    ("Fattahi_Fattahi15", "instances/fattahi/Fattahi15.fjs"),
    ("Hurink_Edata1", "instances/hurink_edata/HurinkEdata1.fjs"),
    ("Hurink_Rdata1", "instances/hurink_rdata/HurinkRdata1.fjs"),
)


def _imports():
    from causal_schedule_lab.ir_adapters.fjsp_drl import load_fjsp_problem
    from causal_schedule_lab.solvers.dispatching import solve_dispatching
    from causal_schedule_lab.symptom_pruning import diagnose_and_prune
    from causal_schedule_lab.legal_edit_enumerator import LegalEditEnumerator
    from causal_schedule_lab.teacher.atomic_counterfactual_executor import (
        AtomicCounterfactualExecutor,
    )
    from causal_schedule_lab.teacher.atom_generator import DecisionAtom
    from causal_schedule_lab.validation import schedule_hash

    return (
        load_fjsp_problem,
        solve_dispatching,
        diagnose_and_prune,
        LegalEditEnumerator,
        AtomicCounterfactualExecutor,
        DecisionAtom,
        schedule_hash,
    )


# -- atom 生成（复制 stage_a 冻结口径 edit_to_atom，不改语义） ----------------
def _edit_to_atom(edit, DecisionAtom):
    if edit.edit_type == "ROUTE":
        return DecisionAtom(
            atom_id=edit.edit_id, atom_type="routing", operation=edit.operation_id,
            machine=edit.target_machine, partner_operation=None,
            probe_operator_id="machine_reassignment",
            probe_parameters={
                "operation_id": edit.operation_id,
                "mode_id": edit.target_mode_id,
                "target_resource_ids": [edit.target_machine],
            },
        )
    if edit.edit_type == "SEQ_SWAP":
        return DecisionAtom(
            atom_id=edit.edit_id, atom_type="sequencing", operation=edit.left_id,
            machine=edit.resource_id, partner_operation=edit.right_id,
            probe_operator_id="adjacent_resource_swap",
            probe_parameters={
                "resource_id": edit.resource_id,
                "left_operation_id": edit.left_id,
                "right_operation_id": edit.right_id,
            },
        )
    if edit.edit_type == "SEQ_INSERT":
        return DecisionAtom(
            atom_id=edit.edit_id, atom_type="sequencing", operation=edit.operation_id,
            machine=edit.resource_id,
            partner_operation=(edit.predecessor_id or edit.successor_id),
            probe_operator_id="resource_sequence_insertion",
            probe_parameters={
                "operation_id": edit.operation_id,
                "resource_id": edit.resource_id,
                "position": edit.insert_position,
                "predecessor_id": edit.predecessor_id,
                "successor_id": edit.successor_id,
            },
        )
    return None


# -- 表象强度口径 -------------------------------------------------------------
def block_strength(block) -> tuple[float, float]:
    """(priority, primary-rule appearance value)."""
    pri = float(getattr(block, "priority", 0.0) or 0.0)
    rules = getattr(block.block, "appearance_rules", ()) or ()
    vals = getattr(block.block, "appearance_values", {}) or {}
    rule_val = float(vals.get(rules[0], 0.0)) if rules else 0.0
    return pri, rule_val


def block_ops(block) -> set[str]:
    return set(getattr(block.block, "operations", ()) or ())


# -- 候选节点：成员 + 同工件直接前驱 + 同机器直接前序（1 跳祖先） --------------
def machine_of(problem, assignment) -> str | None:
    mode = problem.mode_map().get(assignment.mode_id)
    if mode is None:
        return None
    return mode[1].resources[0] if mode[1].resources else None


def block_candidates(problem, schedule, block, *, max_cands: int) -> list[str]:
    op_map = problem.operation_map()
    amap = schedule.assignment_map()
    members = block_ops(block)
    cand: list[str] = []

    # 成员本身
    cand.extend(sorted(members))

    # 同工件直接前驱（precedence 反向）+ 同机器直接前序（resource_sequence 反向）
    for op_id in members:
        op = op_map.get(op_id)
        if op is not None:
            for pred in op.predecessors:
                if pred not in cand:
                    cand.append(pred)
        a = amap.get(op_id)
        if a is None:
            continue
        m = machine_of(problem, a)
        if m is None:
            continue
        # 同机器上 end <= 本 op.start 的最近一个前序
        same_m = [o for o, aa in amap.items() if machine_of(problem, aa) == m and o != op_id]
        before = [o for o in same_m if amap[o].end <= a.start]
        if before:
            latest = max(before, key=lambda o: amap[o].end)
            if latest not in cand:
                cand.append(latest)

    # 去重保序，截断
    seen, out = set(), []
    for c in cand:
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= max_cands:
            break
    return out


# -- 重排后匹配原表象块 -------------------------------------------------------
def match_block_after(after_snap, orig_ops: set[str]):
    """返回 (matched_block | None, best_jaccard)."""
    best, best_j = None, -1.0
    for b in after_snap.retained:
        o = set(getattr(b.block, "operations", ()) or ())
        union = len(orig_ops | o)
        j = len(orig_ops & o) / max(union, 1)
        if j > best_j:
            best, best_j = b, j
    return best, best_j


def _main() -> None:
    (
        load_fjsp_problem, solve_dispatching, diagnose_and_prune,
        LegalEditEnumerator, AtomicCounterfactualExecutor, DecisionAtom,
        schedule_hash,
    ) = _imports()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    executor = AtomicCounterfactualExecutor()

    records: list[dict[str, Any]] = []
    t0 = time.time()

    for instance_id, fjs_path in PROBE_INSTANCES[:N_STATES]:
        problem = load_fjsp_problem(fjs_path, problem_id=instance_id)
        schedule = solve_dispatching(problem, rule="earliest_finish")
        base_ms = int(schedule.makespan)
        base_hash = schedule_hash(schedule)
        state_id = f"S::{instance_id}::{base_hash[:16]}"

        snap = diagnose_and_prune(problem, schedule)
        retained = sorted(snap.retained, key=lambda b: -block_strength(b)[0])
        chosen = retained[:N_BLOCKS_PER_STATE]
        if not chosen:
            continue

        enum = LegalEditEnumerator(problem, schedule)
        all_ops = tuple(sorted(schedule.assignment_map()))
        edits = enum.enumerate(
            all_ops,
            request_route=True, request_seq_swap=True, request_seq_insert=True,
            request_timing_shift=False,
        )
        # op -> 优先 ROUTE，其次 SEQ
        by_op: dict[str, list] = defaultdict(list)
        for e in edits:
            by_op[e.operation_id].append(e)

        for block in chosen:
            bid = block.block.block_id
            pri_before, val_before = block_strength(block)
            orig_ops = block_ops(block)
            cands = block_candidates(problem, schedule, block, max_cands=N_CANDIDATES_PER_BLOCK)

            for op_id in cands:
                # 找该 op 的原子干预：ROUTE 优先，SEQ 兜底
                atom = None
                edit_kind = None
                cand_edits = by_op.get(op_id, [])
                for e in cand_edits:
                    if e.edit_type == "ROUTE":
                        atom, edit_kind = _edit_to_atom(e, DecisionAtom), "ROUTE"
                        break
                if atom is None:
                    for e in cand_edits:
                        a = _edit_to_atom(e, DecisionAtom)
                        if a is not None:
                            atom, edit_kind = a, e.edit_type
                            break
                # 也可能 op 只在 SEQ 里当 partner（left/right），补一个
                if atom is None:
                    for e in edits:
                        if op_id in (getattr(e, "left_id", None), getattr(e, "right_id", None),
                                     getattr(e, "predecessor_id", None), getattr(e, "successor_id", None)):
                            a = _edit_to_atom(e, DecisionAtom)
                            if a is not None:
                                atom, edit_kind = a, e.edit_type
                                break
                if atom is None:
                    records.append({
                        "state_id": state_id, "instance_id": instance_id,
                        "block_id": bid, "op": op_id, "no_atom": True,
                    })
                    continue

                res = executor.execute_atom(problem, schedule, atom)
                if not (res.feasible and res.schedule is not None):
                    records.append({
                        "state_id": state_id, "instance_id": instance_id,
                        "block_id": bid, "op": op_id, "infeasible": True,
                        "edit_kind": edit_kind,
                    })
                    continue

                after_ms = int(res.schedule.makespan)
                after_snap = diagnose_and_prune(problem, res.schedule)
                matched, jac = match_block_after(after_snap, orig_ops)

                if matched is None or jac < 0.5:
                    records.append({
                        "state_id": state_id, "instance_id": instance_id,
                        "block_id": bid, "op": op_id, "edit_kind": edit_kind,
                        "block_disappeared": True, "jaccard": round(jac, 4),
                        "pri_before": pri_before, "val_before": val_before,
                        "makespan_gain": base_ms - after_ms,
                    })
                    continue

                pri_after, val_after = block_strength(matched)
                records.append({
                    "state_id": state_id, "instance_id": instance_id,
                    "block_id": bid, "op": op_id, "edit_kind": edit_kind,
                    "jaccard": round(jac, 4),
                    "pri_before": pri_before, "pri_after": pri_after,
                    "val_before": val_before, "val_after": val_after,
                    "delta_pri": round(pri_before - pri_after, 6),
                    "delta_val": round(val_before - val_after, 6),
                    "makespan_gain": base_ms - after_ms,
                })

    # ---- 汇总 ----
    full = [r for r in records if "delta_pri" in r]
    n_total = len(records)
    n_full = len(full)
    n_noatom = sum(1 for r in records if r.get("no_atom"))
    n_infeas = sum(1 for r in records if r.get("infeasible"))
    n_disapp = sum(1 for r in records if r.get("block_disappeared"))

    delta_pri = [r["delta_pri"] for r in full]
    nz_pri = sum(1 for d in delta_pri if abs(d) > 1e-9)
    nz_val = sum(1 for r in full if abs(r["delta_val"]) > 1e-9)
    pos_pri = sum(1 for d in delta_pri if d > 1e-9)   # 表象变好（priority 降）
    neg_pri = sum(1 for d in delta_pri if d < -1e-9)
    # makespan 与 delta 的一致性：表象变好(delta_pri>0) 且 makespan 也改善(gain>0) 的比例
    both_improve = sum(
        1 for r in full if r["delta_pri"] > 1e-9 and r["makespan_gain"] > 0
    )
    # 每个 block 是否至少有一个候选产生非零 delta
    block_has_signal = defaultdict(set)
    for r in full:
        if abs(r["delta_pri"]) > 1e-9:
            block_has_signal[(r["instance_id"], r["block_id"])].add(r["op"])
    n_blocks = len({(r["instance_id"], r["block_id"]) for r in records})

    summary = {
        "n_states": min(N_STATES, len(PROBE_INSTANCES)),
        "n_blocks": n_blocks,
        "n_records_total": n_total,
        "n_full_attribution": n_full,
        "n_no_atom": n_noatom,
        "n_infeasible": n_infeas,
        "n_block_disappeared": n_disapp,
        "frac_block_disappeared": round(n_disapp / max(n_total, 1), 4),
        "n_nonzero_delta_pri": nz_pri,
        "frac_nonzero_delta_pri": round(nz_pri / max(n_full, 1), 4),
        "n_nonzero_delta_val": nz_val,
        "frac_nonzero_delta_val": round(nz_val / max(n_full, 1), 4),
        "n_pos_pri": pos_pri, "n_neg_pri": neg_pri,
        "n_both_improve": both_improve,
        "frac_blocks_with_signal": round(len(block_has_signal) / max(n_blocks, 1), 4),
        "delta_pri_min": min(delta_pri) if delta_pri else None,
        "delta_pri_max": max(delta_pri) if delta_pri else None,
        "wall_seconds": round(time.time() - t0, 1),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "records.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n"
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
