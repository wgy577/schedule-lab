"""Canonical M3 proposal feature construction + Frozen-Local execution.

VERBATIM extraction from the legacy R-chain (no semantic change):
  - analyze_state / _single_prop_feat / _pair_prop_feat / build_proposal_features /
    state_feature_vec / _execute_step / AnalyzeCache / _edits_for  <- r1
  - proposal_identity / old_base_of                                <- r2
Internal `r1.x` / `r2.x` references were replaced by the canonical module
equivalents (equivalent code, identical numbers).  Upstream (pilot / b52 /
m3util / runmod) comes from :mod:`causal_schedule_lab.m3.upstream`.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict, deque
import numpy as np
import torch

from causal_schedule_lab.intervention import ScheduleGraphView
from causal_schedule_lab.intervention.composite_legality import check_composite_structural_legality
from causal_schedule_lab.m2_v5_schema_v1 import (
    CausalInterventionProposal,
    EDIT_ROUTE,
    EDIT_SEQ_INSERT,
    EDIT_SEQ_SWAP,
)
from causal_schedule_lab.validation import schedule_hash

from .config import (
    B5_EPS,
    PAIR_FEAT_DIM,
    PAIR_TOP_K,
    T2L_PAIR_ROUTE_ATOMS,
    T2L_PAIR_SEQUENCE_ATOMS,
    T2L_PAIRS_PER_FAMILY,
    T2L_PAIR_TOTAL_CAP,
    T2L_ANALYZE_CACHE_ENTRIES,
)
# NOTE: reference ``upstream.<name>`` DYNAMICALLY (never `from .upstream import X`):
# load_upstream() mutates module globals, so a value-snapshot import would bind
# the pre-load None.
from . import upstream


ROOT_APP_MAX_HOPS = 4
ROOT_APP_MAX_SUPPORT = 8
FIXED_MACHINE_FAMILIES = frozenset({"JSP", "FSP", "DJSP"})


def _analysis_heartbeat(stage, case_id, **details):
    """Refine the worker heartbeat while the expensive analyzer is running."""
    directory = os.environ.get("T2M_WORKER_HEARTBEAT_DIR", "")
    if not directory or os.environ.get("T2M_WORKER_HEARTBEAT", "1") == "0":
        return
    try:
        os.makedirs(directory, exist_ok=True)
        pid = os.getpid()
        target = os.path.join(directory, f"worker_{pid}.json")
        row = {}
        if os.path.exists(target):
            with open(target, encoding="utf-8") as stream:
                row = json.load(stream)
        row.update({
            "pid": pid,
            "wall_time": time.time(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": str(stage),
            "case_id": str(case_id),
            **details,
        })
        temporary = target + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(row, stream, sort_keys=True)
        os.replace(temporary, target)
    except Exception:
        return


def _problem_family(problem):
    """Canonical scheduling family used by the hard action-space contract."""
    return str(getattr(problem, "kind", "GENERIC")).strip().upper()


def _filter_family_legal_edits(problem, edits):
    """Remove operators that are undefined for the declared problem family.

    JSP/FSP (and dynamic JSP) have fixed operation-machine assignments.  A
    malformed imported eligibility list must therefore never turn ROUTE into a
    learnable action.  Flexible families still pass through the enumerator's
    per-operation eligible-machine and stage constraints.
    """
    if _problem_family(problem) in FIXED_MACHINE_FAMILIES:
        return [edit for edit in edits if edit.edit_type != EDIT_ROUTE]
    return list(edits)


def selected_root_paths(ast, roots):
    """One shortest structural reverse path per selected root/appearance.

    Reachability is graph evidence, not an identified intervention effect.
    Called only while capturing evaluation visual plans.
    """
    incoming = {}
    for cause, effect in ast.get("trace_causal_edges", ()):
        incoming.setdefault(effect, []).append(cause)
    rows = []
    for block_id, seeds in ast.get("trace_appearance_seeds", ()):
        paths = {seed: [seed] for seed in seeds}
        queue = deque(seeds)
        while queue:
            effect = queue.popleft()
            if len(paths[effect]) - 1 >= ROOT_APP_MAX_HOPS:
                continue
            for cause in sorted(incoming.get(effect, ())):
                if cause not in paths:
                    paths[cause] = paths[effect] + [cause]
                    queue.append(cause)
        for root in roots:
            path = paths.get(f"operation:{root}")
            rows.append({"appearance_id": block_id, "root": str(root),
                         "reachable_within_limit": path is not None,
                         "hops": len(path) - 1 if path else None,
                         "effect_to_cause_path": path or [],
                         "evidence": "structural_reachability_not_causal_effect"})
    return rows


def _bounded_reverse_hops(batch, block_count, max_hops=None):
    """Return ``[B,N]`` shortest cause-distance from each appearance block.

    The v1.3 causal edge orientation is ``cause -> effect``.  Root discovery
    walks it backwards (``effect -> cause``), using exactly the TRUE_LOCAL_G_C
    mask already consumed by the frozen B5 reverse encoder.  This is a cheap
    graph traversal: it performs no model forward and no schedule replay.
    """
    if max_hops is None:
        max_hops = ROOT_APP_MAX_HOPS
    n_nodes = int(batch.node_numeric.shape[0])
    hops = torch.full((int(block_count), n_nodes), -1, dtype=torch.int16)
    if block_count <= 0 or batch.symptom_block_node_index is None:
        return hops
    edge_mask = getattr(batch, "reverse_causal_mask", None)
    if edge_mask is None:
        edge_mask = ((batch.edge_type == 7) | (batch.edge_type == 8))
    src = batch.edge_index[0, edge_mask].detach().cpu().tolist()
    dst = batch.edge_index[1, edge_mask].detach().cpu().tolist()
    incoming = [[] for _ in range(n_nodes)]
    for cause, effect in zip(src, dst):
        incoming[int(effect)].append(int(cause))
    b_idx, n_idx = (x.detach().cpu().tolist()
                    for x in batch.symptom_block_node_index)
    members = [[] for _ in range(int(block_count))]
    for block, node in zip(b_idx, n_idx):
        if 0 <= int(block) < int(block_count):
            members[int(block)].append(int(node))
    for block, seeds in enumerate(members):
        q = deque()
        for node in dict.fromkeys(seeds):
            hops[block, node] = 0
            q.append(node)
        while q:
            effect = q.popleft()
            depth = int(hops[block, effect])
            if depth >= int(max_hops):
                continue
            for cause in incoming[effect]:
                if int(hops[block, cause]) < 0:
                    hops[block, cause] = depth + 1
                    q.append(cause)
    return hops


def _compact_root_appearance_context(ast, ops, max_support=ROOT_APP_MAX_SUPPORT):
    """Build compact root×appearance relation rows for the M2 actor.

    Full ``h_a[B,N,H]`` remains only in the worker-local AnalyzeCache.  Rollout
    records receive at most ``max_support`` rows per root, preventing the new
    causal context from multiplying IPC/RAM by trajectories × steps.
    """
    h_a = ast.get("appearance_h_a")
    q_a = ast.get("appearance_q")
    scores = ast.get("per_block_candidate_scores")
    hops = ast.get("root_appearance_hops")
    app_features = ast.get("appearance_features")
    n_root = len(ops)
    hidden = int(ast["h_c"].shape[-1])
    app_feat_dim = (int(app_features.shape[-1]) if app_features is not None and
                    app_features.dim() == 2 else 0)
    # q_A + h_a(root|A) + score + hop_norm + hop0 + reachable + app features
    raw_dim = 2 * hidden + 4 + app_feat_dim
    raw = torch.zeros((n_root, int(max_support), raw_dim), dtype=torch.float32)
    mask = torch.zeros((n_root, int(max_support)), dtype=torch.bool)
    min_hop = torch.full((n_root,), -1, dtype=torch.int16)
    support_count = torch.zeros((n_root,), dtype=torch.int16)
    support_blocks = []
    if h_a is None or q_a is None or scores is None or len(q_a) == 0:
        return {"raw": raw, "mask": mask, "raw_dim": raw_dim,
                "min_hop": min_hop, "support_count": support_count,
                "support_blocks": tuple(() for _ in ops)}
    for ri, op in enumerate(ops):
        ni = ast["node_index"].get(f"operation:{op}")
        if ni is None:
            support_blocks.append(())
            continue
        rows = []
        for bi in range(int(scores.shape[0])):
            hop = int(hops[bi, ni]) if hops is not None else -1
            score = float(scores[bi, ni])
            reachable = hop >= 0
            # Preserve every structurally reachable appearance.  A scored but
            # non-reachable row is also retained as model evidence, explicitly
            # marked reachable=0 rather than pretending it has a causal hop.
            if not reachable and score <= float(B5_EPS):
                continue
            rows.append((bi, hop, score, reachable))
        rows.sort(key=lambda x: (not x[3], x[1] if x[1] >= 0 else 999,
                                 -x[2], x[0]))
        # L's support count is a scientific metric: number of distinct
        # appearances that are structurally reachable from this intervention
        # node.  B5-only scored evidence remains available to attention below,
        # but can no longer masquerade as a shared upstream control relation.
        reachable_rows = [row for row in rows if row[3]]
        support_count[ri] = len(reachable_rows)
        finite = [row[1] for row in rows if row[1] >= 0]
        min_hop[ri] = min(finite) if finite else -1
        chosen = rows[:int(max_support)]
        if ast.get("_e2e_input") is not None:
            ast.setdefault("_e2e_app_rows", {})[op] = [row[0] for row in chosen]
        support_blocks.append(tuple(int(row[0]) for row in reachable_rows[:int(max_support)]))
        for si, (bi, hop, score, reachable) in enumerate(chosen):
            fields = [q_a[bi].float(), h_a[bi, ni].float(),
                      torch.tensor([score,
                                    (float(hop) / max(ROOT_APP_MAX_HOPS, 1)
                                     if hop >= 0 else 1.0),
                                    float(hop == 0), float(reachable)])]
            if app_feat_dim:
                fields.append(app_features[bi].float())
            raw[ri, si] = torch.cat(fields)
            mask[ri, si] = True
    return {"raw": raw.detach(), "mask": mask, "raw_dim": raw_dim,
            "min_hop": min_hop, "support_count": support_count,
            "support_blocks": tuple(support_blocks)}


def _edit_signature(e):
    """Stable identity for every executable edit family.

    Keep the historical ROUTE signature byte-for-byte compatible with the SFT
    checkpoints and Memory records.  Sequence edits need their resource/slot
    identity; using the old ``source_machine->target_machine`` spelling would
    collapse every sequence edit for an operation to ``None->None``.
    """
    if e.edit_type == EDIT_ROUTE:
        return upstream.b52._sig(e)
    if e.edit_type == EDIT_SEQ_SWAP:
        return (f"{e.edit_type}:{e.left_id}<->{e.right_id}"
                f"|{e.resource_id}")
    if e.edit_type == EDIT_SEQ_INSERT:
        return (f"{e.edit_type}:{e.operation_id}@{e.resource_id}"
                f":pos{e.insert_position}:{e.predecessor_id}>{e.successor_id}")
    return str(e.edit_id)


def _sequence_root_ops(e, attributed_ops):
    """Causal roots touched by one sequence edit, in stable order."""
    attributed = set(attributed_ops)
    if e.edit_type == EDIT_SEQ_SWAP:
        touched = (e.left_id, e.right_id)
    else:
        touched = (e.operation_id,)
    roots = tuple(op for op in touched if op is not None and op in attributed)
    return roots or tuple(op for op in touched if op is not None)


def _augment_sequence_contributors(enum, op_b5, existing_edits, graph_view=None):
    """Add bounded precedence-safe sequencing moves for attributed roots.

    The frozen B5.2 path was route-only.  That silently leaves JSP/FSP with no
    action even when M2 found a valid causal operation.  The canonical legal
    enumerator and D6 bridge already support adjacent swaps and bounded inserts,
    so expose those *single* actions without changing attribution or rewards.
    """
    roots = tuple(sorted(op_b5))
    if not roots:
        return []
    seq = enum.enumerate(
        roots,
        request_route=False,
        request_seq_swap=True,
        request_seq_insert=True,
        request_timing_shift=False,
    )
    seen = {str(e.edit_id) for e in existing_edits}
    candidates = [e for e in seq
                  if e.edit_type in (EDIT_SEQ_SWAP, EDIT_SEQ_INSERT)
                  and str(e.edit_id) not in seen]
    if graph_view is None:
        return candidates
    return [e for e in candidates
            if check_composite_structural_legality(graph_view, (e,)).legal]


# ---------------------------------------------------------------------------
# state reconstruction for arbitrary (successor) schedules  [r1]
# ---------------------------------------------------------------------------
def analyze_state(problem, schedule, model_b5, case_id):
    """Re-diagnose + re-run M2 (B5) for an arbitrary schedule -> h_c + op_b5 + pool."""
    t_all = time.perf_counter()
    _analysis_heartbeat("diagnose_and_prune", case_id)
    appearance = upstream.pilot.diagnose_and_prune(problem, schedule).model_dump(mode="json")
    _analysis_heartbeat("compile_sg_sct", case_id,
                        appearance_blocks=len(appearance.get("blocks", ())))
    bundle = upstream.pilot.compile_sg_sct_input_v1_3(problem, schedule, appearance, case_id=case_id)
    node_ids = tuple(bundle.manifest["id_spaces"]["node_ids"])
    block_ids = tuple(bundle.manifest["id_spaces"]["appearance_block_ids"])
    context = upstream.pilot.build_m2_runtime_context(problem, schedule, appearance, block_ids=block_ids)
    batch = upstream.pilot.to_sg_sct_batch_v1_3(bundle, device="cpu")
    c_prior = upstream.pilot.compute_c_prior(problem, schedule, node_ids)
    t_neural = time.perf_counter()
    _analysis_heartbeat("frozen_b5_forward", case_id,
                        graph_nodes=len(node_ids), appearance_blocks=len(block_ids))
    model_b5.bind_runtime_context(context, node_ids=list(node_ids))
    with torch.no_grad():
        encoded = model_b5._encode_h_a(batch)
        out = model_b5.attribution_forward(batch, c_prior, encoded=encoded)
        h_a, h_c, q_A, block_count = encoded
    neural_s = time.perf_counter() - t_neural
    t_post = time.perf_counter()
    _analysis_heartbeat("proposal_pool_build", case_id,
                        graph_nodes=len(node_ids), appearance_blocks=len(block_ids))
    scores = out.per_block_candidate_score.numpy()
    reverse_hops = _bounded_reverse_hops(batch, block_count)
    trace_mask = getattr(batch, "reverse_causal_mask", None)
    if trace_mask is None:
        trace_mask = ((batch.edge_type == 7) | (batch.edge_type == 8))
    trace_edges = tuple((node_ids[s], node_ids[t]) for s, t in
                        batch.edge_index[:, trace_mask].T.tolist())
    trace_seeds = [[] for _ in block_ids]
    if batch.symptom_block_node_index is not None:
        for bi, ni in batch.symptom_block_node_index.T.tolist():
            trace_seeds[bi].append(node_ids[ni])
    appearance_features = (batch.appearance_features.detach().cpu().float()
                           if batch.appearance_features is not None else None)
    node_index = {n: i for i, n in enumerate(node_ids)}
    # op_b5 over ALL retained blocks (makespan objective -> no teacher `important` filter)
    op_b5: dict[str, float] = {}
    op_n_blocks: dict[str, int] = {}
    for bi in range(scores.shape[0]):
        for ni in np.argsort(-scores[bi]):
            s = float(scores[bi, ni])
            if s <= B5_EPS:
                break
            nid = node_ids[ni]
            if nid.startswith("operation:"):
                op = nid[len("operation:"):]
                op_b5[op] = max(op_b5.get(op, 0.0), s)
                op_n_blocks[op] = op_n_blocks.get(op, 0) + 1
    enum, contrib, enab, role = upstream.b52.build_pool(problem, schedule, op_b5)
    # Family applicability is a hard action-space constraint, not something
    # the policy should waste samples learning.  This also protects JSP/FSP
    # against incorrectly imported multi-machine eligibility metadata.
    contrib = _filter_family_legal_edits(problem, contrib)
    enab = _filter_family_legal_edits(problem, enab)
    graph_view = ScheduleGraphView.from_problem_schedule(problem, schedule)
    sequence = _augment_sequence_contributors(enum, op_b5,
                                               list(contrib) + list(enab),
                                               graph_view=graph_view)
    for e in sequence:
        role[_edit_signature(e)] = "CONTRIBUTOR"
    pool = []
    for e in list(contrib) + list(enab) + sequence:
        # LegalEditEnumerator guarantees local edit well-formedness.  This
        # whole-schedule DAG check additionally guarantees that job precedence
        # plus the edited machine order remains acyclic before M2/M3 can sample
        # the action.  It applies to ROUTE as well as both sequence operators.
        if not check_composite_structural_legality(graph_view, (e,)).legal:
            continue
        sig = _edit_signature(e)
        root_ops = (_sequence_root_ops(e, op_b5) if e.edit_type != EDIT_ROUTE
                    else (e.operation_id,))
        embedding_op = (e.operation_id if f"operation:{e.operation_id}" in node_index
                        else next((op for op in root_ops
                                   if f"operation:{op}" in node_index), None))
        ni = node_index.get(f"operation:{embedding_op}")
        if ni is None:
            continue
        feat = torch.tensor(upstream.m3util.single_feature_vec(e, op_b5, role[sig]), dtype=torch.float32)
        # For a swap whose left operation is not itself attributed, carry the
        # strongest touched-root attribution in the existing B5 feature slot.
        if e.edit_type != EDIT_ROUTE and root_ops:
            feat[10] = max(float(op_b5.get(op, 0.0)) for op in root_ops)
        pool.append({"e": e, "sig": sig, "ni": ni, "feat": feat,
                     "role": role[sig], "h": h_c[ni],
                     "root_ops": tuple(root_ops)})
    post_s = time.perf_counter() - t_post
    total_s = time.perf_counter() - t_all
    _analysis_heartbeat("analysis_complete", case_id,
                        proposals=len(pool), duration_s=round(total_s, 6))
    family = _problem_family(problem)
    return {"_e2e_input": ({"batch": batch, "context": context,
             "node_ids": node_ids, "c_prior": c_prior}
             if getattr(model_b5, "e2e_capture", False) else None),
            "problem": problem, "schedule": schedule, "appearance": appearance,
            "problem_family": family,
            "route_operator_allowed": family not in FIXED_MACHINE_FAMILIES,
            "h_c": h_c, "op_b5": op_b5,
            "op_n_blocks": op_n_blocks, "node_index": node_index, "enum": enum,
            "graph_view": graph_view,
            # Worker-local full tensors.  `_root_candidate_arrays` immediately
            # compresses these to bounded root×appearance rows before rollout
            # records cross a process boundary.
            "appearance_block_ids": block_ids,
            "trace_causal_edges": trace_edges,
            "trace_appearance_seeds": tuple(zip(block_ids, trace_seeds)),
            "appearance_q": (q_A.detach().cpu().float() if q_A is not None else
                             torch.zeros((0, h_c.shape[-1]), dtype=torch.float32)),
            "appearance_h_a": h_a.detach().cpu().float(),
            "appearance_features": appearance_features,
            "per_block_candidate_scores": torch.as_tensor(scores).float(),
            "root_appearance_hops": reverse_hops,
            "pool": pool, "contrib": sum(rec["role"] == "CONTRIBUTOR"
                                            for rec in pool),
            "enab": sum(rec["role"] == "ENABLER" for rec in pool),
            "operator_counts": {
                EDIT_ROUTE: sum(rec["e"].edit_type == EDIT_ROUTE for rec in pool),
                EDIT_SEQ_SWAP: sum(rec["e"].edit_type == EDIT_SEQ_SWAP for rec in pool),
                EDIT_SEQ_INSERT: sum(rec["e"].edit_type == EDIT_SEQ_INSERT for rec in pool),
            },
            "_profile": {"graph_construction_s": max(total_s - neural_s - post_s, 0.0),
                         "neural_forward_s": neural_s, "other_analysis_s": post_s,
                         "total_s": total_s}}


def _single_prop_feat(rec):
    h_u = rec["h"]
    h_v = torch.zeros_like(h_u)
    feat_u = rec["feat"]
    feat_v = torch.zeros_like(feat_u)
    pstruct = torch.zeros(PAIR_FEAT_DIM)
    uhat = rec["uhat"]
    return torch.cat([h_u, h_v, feat_u, feat_v, pstruct,
                      torch.tensor([uhat, 0.0, uhat, 0.0], dtype=torch.float32)])


def _pair_prop_feat(u, v, pstruct_t, direct):
    return torch.cat([u["h"], v["h"], u["feat"], v["feat"], pstruct_t,
                      torch.tensor([u["uhat"], v["uhat"], direct, 1.0], dtype=torch.float32)])


def _pair_family(u, v):
    """Exact two-atom family used for independent quotas and diagnostics."""
    order = {EDIT_ROUTE: 0, EDIT_SEQ_SWAP: 1, EDIT_SEQ_INSERT: 2}
    kinds = [str(rec["e"].edit_type) for rec in (u, v)]
    kinds.sort(key=lambda kind: (order.get(kind, 99), kind))
    return "+".join(kinds)


def _touched_operations(edit):
    if edit.edit_type == EDIT_SEQ_SWAP:
        return {op for op in (edit.left_id, edit.right_id) if op is not None}
    return {edit.operation_id}


def _joint_operator_contract_compatible(u, v):
    """Cheap fail-closed mirror of executor multi-atom interaction rules.

    Structural DAG legality alone is insufficient.  A ROUTE edit changes the
    membership of both its source and target machine, invalidating any sequence
    edit compiled against either observed sequence.  Likewise, two independently
    enumerated sequence edits on one observed machine can invalidate each other's
    neighbor/slot contract after the first edit.  Reject those compositions before
    M2/M3 see them; cross-machine combinations remain available.
    """
    edits = (u["e"], v["e"])
    routes = [edit for edit in edits if edit.edit_type == EDIT_ROUTE]
    sequences = [edit for edit in edits if edit.edit_type != EDIT_ROUTE]
    changed_machines = {
        machine
        for edit in routes
        for machine in (edit.source_machine, edit.target_machine)
        if machine is not None
    }
    if any(edit.resource_id in changed_machines for edit in sequences):
        return False, "routing_sequence_contract"
    sequence_resources = [edit.resource_id for edit in sequences]
    if len(sequence_resources) != len(set(sequence_resources)):
        return False, "same_resource_sequence_contract"
    return True, "ok"


def build_proposal_features(ast, single_head, direct_head, *, fast_pairs=False):
    """Return (prop_feats [N,300] | None, metas, agg)."""
    pool = ast["pool"]
    if not pool:
        return None, [], {"best_uhat": 0.0, "best_direct": 0.0, "n_contrib": 0, "n_enab": 0}
    # single utility predictions (batched)
    hs = torch.stack([rec["h"] for rec in pool])
    fs = torch.stack([rec["feat"] for rec in pool])
    with torch.no_grad():
        uhs = single_head(torch.cat([hs, fs], dim=-1)).numpy()
    for idx, rec in enumerate(pool):
        rec["uhat"] = float(uhs[idx])

    feats = []
    metas = []
    best_uhat = -1e18
    best_direct = -1e18
    for idx, rec in enumerate(pool):
        f = _single_prop_feat(rec)
        feats.append(f)
        metas.append({"kind": "single", "i": idx, "j": None})
        best_uhat = max(best_uhat, rec["uhat"])
        best_direct = max(best_direct, rec["uhat"])

    # Pair enumeration (composite-legal, bounded by atomic policy priors).
    #
    # The inherited code considered ROUTE+ROUTE only, despite downstream logs
    # claiming ROUTE+SEQ and SEQ+SEQ coverage.  Joint-search experiments found
    # useful route+swap, route+insert and swap+swap actions, so all three coarse
    # families are materialised here.  We first retain a small ranked atomic
    # set, avoiding an O(N^2) legality pass over the full proposal pool.
    gv = ast.get("graph_view") or ScheduleGraphView.from_problem_schedule(
        ast["problem"], ast["schedule"])
    ranked_routes = sorted(
        (i for i, rec in enumerate(pool) if rec["e"].edit_type == EDIT_ROUTE),
        key=lambda i: (-float(pool[i]["uhat"]), pool[i]["sig"]))[
            :max(int(T2L_PAIR_ROUTE_ATOMS), 0)]
    ranked_sequences = sorted(
        (i for i, rec in enumerate(pool) if rec["e"].edit_type != EDIT_ROUTE),
        key=lambda i: (-float(pool[i]["uhat"]), pool[i]["sig"]))[
            :max(int(T2L_PAIR_SEQUENCE_ATOMS), 0)]
    bounded = ranked_routes + ranked_sequences
    # Five evidence-backed two-atom families.  INSERT+INSERT is deliberately
    # omitted: it was not supported by the joint-search experiment and causes
    # a disproportionately large positional Cartesian product.
    pair_families = (
        "ROUTE+ROUTE",
        "ROUTE+SEQ_SWAP",
        "ROUTE+SEQ_INSERT",
        "SEQ_SWAP+SEQ_SWAP",
        "SEQ_SWAP+SEQ_INSERT",
    )
    per_family = {family: [] for family in pair_families}
    if fast_pairs:
        from .fast_inference import companion_pairs
        fast_candidates = companion_pairs(pool, gv, partner_scan=8)
    else:
        fast_candidates = None
    for ii, i in enumerate(bounded):
        if fast_candidates is not None:
            break
        for j in bounded[ii + 1:]:
            u, v = pool[i], pool[j]
            if _touched_operations(u["e"]) & _touched_operations(v["e"]):
                continue
            compatible, _reason = _joint_operator_contract_compatible(u, v)
            if not compatible:
                continue
            leg = check_composite_structural_legality(gv, (u["e"], v["e"]))
            if not leg.legal:
                continue
            family = _pair_family(u, v)
            if family not in per_family:
                continue
            add = float(u["uhat"]) + float(v["uhat"])
            per_family[family].append((add, i, j, family))
    if fast_candidates is not None:
        for i, j in fast_candidates:
            family = _pair_family(pool[i], pool[j])
            per_family[family].append((float(pool[i]['uhat']) +
                                      float(pool[j]['uhat']), i, j, family))
    cand = []
    family_cap = max(int(T2L_PAIRS_PER_FAMILY), 0)
    for family in pair_families:
        rows = sorted(per_family[family], key=lambda x: (-x[0], x[1], x[2]))
        cand.extend(rows[:family_cap])
    cand.sort(key=lambda x: (-x[0], x[3], x[1], x[2]))
    cand = cand[:min(max(int(T2L_PAIR_TOTAL_CAP), 0), int(PAIR_TOP_K))]
    enum = ast["enum"]
    problem, schedule = ast["problem"], ast["schedule"]
    pair_counts = {family: 0 for family in pair_families}
    for (_, i, j, family) in cand:
        u, v = pool[i], pool[j]
        # The frozen direct head was trained only for route pairs.  Mixed and
        # sequence pairs therefore use a neutral structural vector and the
        # additive frozen single prior; the trainable M3 residual learns their
        # actual net value from GRPO instead of consuming out-of-domain logits.
        if family == "ROUTE+ROUTE":
            pstruct = upstream.b52.pair_feature_vec(
                problem, schedule, enum,
                u["e"].operation_id, u["e"].source_machine, u["e"].target_machine,
                v["e"].operation_id, v["e"].source_machine, v["e"].target_machine,
            )
        else:
            pstruct = [0.0] * PAIR_FEAT_DIM
        pstruct_t = torch.tensor(pstruct, dtype=torch.float32)
        if family == "ROUTE+ROUTE":
            Xd = torch.cat([u["h"], v["h"], u["feat"], v["feat"], pstruct_t]).unsqueeze(0)
            with torch.no_grad():
                direct = float(direct_head(Xd).item())
        else:
            direct = float(u["uhat"] + v["uhat"])
        feats.append(_pair_prop_feat(u, v, pstruct_t, direct))
        metas.append({"kind": "pair", "i": i, "j": j,
                      "family": family})
        pair_counts[family] += 1
        best_direct = max(best_direct, direct)

    if not feats:
        return None, [], {"best_uhat": 0.0, "best_direct": 0.0, "n_contrib": ast["contrib"],
                          "n_enab": ast["enab"]}
    prop_feats = torch.stack(feats)
    agg = {"best_uhat": float(best_uhat), "best_direct": float(best_direct),
           "n_contrib": ast["contrib"], "n_enab": ast["enab"],
           "pair_family_counts": pair_counts}
    return prop_feats, metas, agg


def state_feature_vec(ms, s0_ms, n_prop, best_uhat, best_direct, n_contrib, n_enab):
    return [ms / max(s0_ms, 1.0), (s0_ms - ms) / max(s0_ms, 1.0),
            min(n_prop, 500) / 500.0, best_uhat / max(s0_ms, 1.0),
            best_direct / max(s0_ms, 1.0), min(n_contrib, 200) / 200.0,
            min(n_enab, 500) / 500.0]


# ---------------------------------------------------------------------------
# execution (Frozen-Local) -- returns successor schedule  [r1]
# ---------------------------------------------------------------------------
def _execute_step_with_reason(executor, problem, schedule, edits, base_ms, base_hash):
    """Execute one proposal and retain the fail-closed reason for diagnostics."""
    prop = CausalInterventionProposal(
        proposal_id="rl", appearance_id="rl",
        edits=tuple(edits), root_path=tuple(e.operation_id for e in edits),
    )
    atoms = upstream.runmod._d6_proposal_to_atoms(problem, schedule, prop)
    if atoms is None:
        return None, "unmappable_edit_fail_closed"
    if schedule_hash(schedule) != base_hash:
        raise SystemExit("BLOCKER: executor mutated baseline")
    res = upstream.runmod._d6_execute(executor, problem, schedule, atoms)
    if not res.feasible or res.schedule is None:
        return None, str(getattr(res.report, "reason", None) or "infeasible")
    succ_ms = int(res.schedule.makespan)
    return ({"feasible": True, "improvement": base_ms - succ_ms,
             "schedule": res.schedule}, "ok")


def _execute_step(executor, problem, schedule, edits, base_ms, base_hash):
    result, _reason = _execute_step_with_reason(
        executor, problem, schedule, edits, base_ms, base_hash)
    return result


# ---------------------------------------------------------------------------
# analyze cache (by schedule hash) -- successor states are deterministic  [r1]
# ---------------------------------------------------------------------------
class AnalyzeCache:
    """Bounded LRU for expensive state analysis/proposal materialisation.

    Rollout siblings rapidly diverge, so an unbounded per-worker cache retains
    many large graph/schedule tensors with little reuse and can trigger cgroup
    OOM long before Python reports a useful exception.  Eviction is semantics-
    preserving because both cached functions are deterministic pure analyses.
    """

    def __init__(self, model_b5, single_head, direct_head, max_entries=None):
        self.model_b5 = model_b5
        self.single_head = single_head
        self.direct_head = direct_head
        self.max_entries = max(1, int(
            T2L_ANALYZE_CACHE_ENTRIES if max_entries is None else max_entries))
        self._ast = OrderedDict()
        self._prop = OrderedDict()
        self.ast_calls = 0
        self.ast_misses = 0
        self.ast_seconds = 0.0
        self._ast_seconds_by_key = OrderedDict()

    def _trim(self, mapping, *, on_evict=None):
        while len(mapping) > self.max_entries:
            key, _value = mapping.popitem(last=False)
            if on_evict is not None:
                on_evict(key)

    def ast(self, problem, schedule, iid):
        self.ast_calls += 1
        h = schedule_hash(schedule)
        key = f"{iid}::{h}"
        if key not in self._ast:
            started = time.perf_counter()
            self._ast[key] = analyze_state(
                problem, schedule, self.model_b5, case_id=f"S::{iid}::{h}")
            elapsed = time.perf_counter() - started
            self.ast_misses += 1
            self.ast_seconds += elapsed
            self._ast_seconds_by_key[key] = elapsed
            self._ast_seconds_by_key.move_to_end(key)
            self._trim(
                self._ast,
                on_evict=lambda old: self._ast_seconds_by_key.pop(old, None))
            self._trim(self._ast_seconds_by_key)
        else:
            self._ast.move_to_end(key)
            if key in self._ast_seconds_by_key:
                self._ast_seconds_by_key.move_to_end(key)
        return self._ast[key]

    def ast_seconds_for(self, schedule, iid):
        key = f"{iid}::{schedule_hash(schedule)}"
        return float(self._ast_seconds_by_key.get(key, 0.0))

    def ast_breakdown_for(self, schedule, iid):
        key = f"{iid}::{schedule_hash(schedule)}"
        ast = self._ast.get(key, {})
        if key in self._ast:
            self._ast.move_to_end(key)
        return dict(ast.get("_profile", {}))

    def proposals(self, problem, schedule, iid):
        h = schedule_hash(schedule)
        key = f"{iid}::{h}"
        if key not in self._prop:
            ast = self.ast(problem, schedule, iid)
            self._prop[key] = build_proposal_features(
                ast, self.single_head, self.direct_head)
            self._trim(self._prop)
        else:
            self._prop.move_to_end(key)
        return self._prop[key]


# ---------------------------------------------------------------------------
# edits for a proposal meta  [r1]
# ---------------------------------------------------------------------------
def _edits_for(ast, meta):
    pool = ast["pool"]
    if meta["kind"] == "single":
        return [pool[meta["i"]]["e"]], "single"
    return [pool[meta["i"]]["e"], pool[meta["j"]]["e"]], "pair"


# ---------------------------------------------------------------------------
# proposal identity / frozen base  [r2]
# ---------------------------------------------------------------------------
def proposal_identity(ast, meta):
    """Return (edits, kind, sig, role, type, src, tgt, op_ids)."""
    pool = ast["pool"]
    if meta["kind"] == "single":
        i = meta["i"]
        e = pool[i]["e"]
        role = pool[i]["role"]
        if e.edit_type == EDIT_SEQ_SWAP:
            touched = [op for op in (e.left_id, e.right_id) if op is not None]
            return ([e], "single", pool[i]["sig"], role, "single",
                    e.resource_id, e.resource_id, touched)
        if e.edit_type == EDIT_SEQ_INSERT:
            return ([e], "single", pool[i]["sig"], role, "single",
                    e.resource_id, e.resource_id, [e.operation_id])
        return ([e], "single", pool[i]["sig"], role, "single",
                e.source_machine, e.target_machine, [e.operation_id])
    i, j = meta["i"], meta["j"]
    u, v = pool[i]["e"], pool[j]["e"]
    ru, rv = pool[i]["role"], pool[j]["role"]
    rr = "CC" if (ru == "CONTRIBUTOR" and rv == "CONTRIBUTOR") else \
         "EE" if (ru == "ENABLER" and rv == "ENABLER") else "CE"
    sig = f"{pool[i]['sig']}||{pool[j]['sig']}"
    return ([u, v], "pair", sig, rr, "pair", "", "", [u.operation_id, v.operation_id])


def old_base_of(prop_feats, k):
    """frozen utility ranking: single -> uhat(idx296), pair -> direct(idx298)."""
    return float(prop_feats[k, 296]) if prop_feats[k, 299] < 0.5 else float(prop_feats[k, 298])
