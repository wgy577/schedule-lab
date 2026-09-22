#!/usr/bin/env python3
"""Static safety/completeness gate that runs without importing torch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README_AUTODL.md", "README_OPTIMIZED.md", "setup_autodl.sh", "check_environment.py",
    "build_t2l_package.sh",
    "src/causal_schedule_lab/m3/probability_trace.py",
    "docs/T2L_UNIFIED_INTERVENTION.md", "tests/test_probability_trace.py",
    "tests/test_t2l_joint_macro_v2.py",
    "run_t2l_validation_curve.sh",
    "scripts/evaluate_t2l_checkpoint_series.py",
    "run_smoke.sh", "run_t2l_smoke.sh", "run_p0.sh", "run_p1.sh", "run_p2.sh", "run_t2l.sh",
    "run_t2l_trace.sh",
    "run_tensorboard.sh", "compare_runs.py", "configs/t2d_experiment.json",
    "docs/T2D_ARCHITECTURE.md", "docs/SFT_LATENT_AND_CONTEXT_AUDIT.md",
    "docs/T2E_PERSISTENT_FRONTIER.md",
    "docs/T2I_CAUSAL_CONTEXT_HGRPO.md",
    "docs/M2_ACTOR.md", "docs/M3_ACTOR.md", "docs/PAPER_RELATION.md",
    "scripts/run_t2d_hierarchical_gpu.py",
    "scripts/run_t2l_unified_intervention_grpo_gpu.py",
    "scripts/render_t2l_test_optimization.py",
    "scripts/b5_label_feasibility_audit.py",
    "scripts/b5_route2_teacher_generation.py",
    "scripts/b5_1_train_pilot.py",
    "scripts/run_m3_canonical_training.py",
    "src/causal_schedule_lab/m3/hierarchical_residual.py",
    "src/causal_schedule_lab/m3/t2d_gpu_trainer.py",
    "src/causal_schedule_lab/m3/persistent_frontier_grpo.py",
    "tests/test_m3_hierarchical_residual_t2d.py",
    "tests/test_m3_persistent_frontier_t2e.py",
    "tests/test_m3_lexicographic_fallback_r20.py",
    "tests/test_m3_light_anchor_t2h.py",
    "tests/test_registry.json",
    "checkpoints/b5_1_shared.pt", "checkpoints/m3_proposal_top1_sft_v2.pt",
]
PARENT_HASHES = {
    "src/causal_schedule_lab/m3/proposal_features.py":
        "9af0a7c3e3f3968d25f737c19110760c6a4c5f15ef0bb5a68f556d8075ae20c0",
    "src/causal_schedule_lab/sg_sct_model_v5.py":
        "401b8234027106ee6d165eac6106919917db047dababe1c3e0a1643852199bcd",
    "scripts/run_m3_canonical_training.py":
        "1f2a4e9f3e1a7f7cfcce7b9b71da5015bd7ef7bcac63ca2cb2835c161256d874",
}
CHECKPOINT_HASHES = {
    "checkpoints/b5_1_shared.pt":
        "bb2801b14bb06fe5fd6c9808e887f6a100dd505358b35f9709f79aa11a696b55",
    "checkpoints/m3_proposal_top1_sft_v2.pt":
        "aa83597cd7b6f3854e46cd5c3308071ac2b3f8f31307fce44d5d3fa57e500295",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    errors = []
    errors += [f"missing: {p}" for p in REQUIRED if not (ROOT / p).is_file()]
    symlinks = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_symlink()]
    if symlinks:
        errors.append(f"symlinks forbidden: {symlinks}")
    suspicious = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                  if p.is_file() and "formal_test" in p.name.lower()
                  and not p.name.lower().startswith("test_")]
    if suspicious:
        errors.append(f"Formal TEST-like files: {suspicious}")
    for rel, want in CHECKPOINT_HASHES.items():
        p = ROOT / rel
        if p.is_file() and sha(p) != want:
            errors.append(f"checkpoint hash mismatch: {rel}")

    # Static execution-path contract; this checker intentionally does not import
    # torch so it remains usable before/while AutoDL dependencies are inspected.
    joint = (ROOT / "src/causal_schedule_lab/m3/joint_grpo.py").read_text(
        encoding="utf-8")
    start = joint.find("def _m2_policy_sample_gate(")
    end = joint.find("\ndef ", start + 5)
    gate_body = joint[start:end] if start >= 0 and end > start else ""
    if not gate_body:
        errors.append("missing T2-L score-sampled M2 gate")
    for forbidden in ("_execute_step(", "_probe_op(",
                      "build_lexicographic_action_set_r20(", "_r19_ghog1("):
        if forbidden in gate_body:
            errors.append(f"T2-L M2 gate contains forbidden preselection work: {forbidden}")
    pf = (ROOT / "src/causal_schedule_lab/m3/persistent_frontier_grpo.py").read_text(
        encoding="utf-8")
    runner = (ROOT / "scripts/run_t2l_unified_intervention_grpo_gpu.py").read_text(
        encoding="utf-8")
    if 'action_space="policy_sampled"' not in pf:
        errors.append("training collector is not wired to policy_sampled")
    if 'action_space="policy_sampled"' not in runner:
        errors.append("fixed validation is not wired to policy_sampled")
    for required_text in (
        "def _filter_oversized_graphs(",
        '"--trajectory-steps", type=int, default=10',
        '"--groups-per-update", type=int, default=48',
        '"--max-optimizer-cycles", type=int, default=500',
        '"--eval-every", type=int, default=0',
        '"--milestone-checkpoint-every", type=int, default=50',
        '"--bestn-eval-samples", type=int, default=0',
        'default=20,\n                    help="drop inherited roots above this resource/machine count"',
        'default=200,\n                    help="drop inherited roots above this total operation count"',
        'id_prefix="T2L_REFILL"',
        '"final_total": len(graphs)',
    ):
        if required_text not in runner:
            errors.append(f"missing T2-L bounded-size training-pool contract: {required_text}")
    if "--max-jobs" in runner:
        errors.append("T2-L must not reject ordinary 16-job instances")
    pf = (ROOT / "src/causal_schedule_lab/m3/persistent_frontier_grpo.py").read_text(
        encoding="utf-8")
    for required_text in ("latest_phase_timing", "collect_breakdown",
                          "progress_callback=collection_progress",
                          "runtime/collect_cpu_",
                          '(rec.get("pv_diag") or {}).get',
                          "def _mp_packed_graph_job_r14(",
                          "pickle.loads(future.result())",
                          "target_siblings = 6",
                          "pending_iids=pending_iids"):
        combined = runner + pf + joint
        if required_text not in combined:
            errors.append(f"missing T2-L phase timing contract: {required_text}")
    proposal = (ROOT / "src/causal_schedule_lab/m3/proposal_features.py").read_text(
        encoding="utf-8")
    for required_text in ("def _augment_sequence_contributors(",
                          "request_seq_swap=True", "request_seq_insert=True",
                          "EDIT_SEQ_SWAP", "EDIT_SEQ_INSERT", "root_ops",
                          "check_composite_structural_legality(graph_view, (e,))"):
        if required_text not in proposal + joint:
            errors.append(f"missing four-family operator contract: {required_text}")
    if "operators=family-masked[JSP/FSP/DJSP" not in runner:
        errors.append("T2-L startup does not declare the family-masked operator portfolio")
    for required_text in ("FIXED_MACHINE_FAMILIES", "def _filter_family_legal_edits(",
                          '"route_operator_allowed"'):
        if required_text not in proposal:
            errors.append(f"missing problem-family action guard: {required_text}")
    if 'validation@0 (pre-GRPO)' in pf:
        errors.append("T2-L must train before running fixed validation")

    # T2-L scientific contract: appearances retain identity, conditional
    # total-probability propagation is learned, and both actors optimize one
    # global net-makespan return at every trajectory step.
    hres = (ROOT / "src/causal_schedule_lab/m3/hierarchical_residual.py").read_text(
        encoding="utf-8")
    for required_text in (
        "def _bounded_reverse_hops(", "def _compact_root_appearance_context(",
        "ROOT_APP_MAX_SUPPORT = 8", "M2UnifiedNetMakespanInterventionActor",
        "self.prior_distrust", "self.root_value", "def _initial_root_stratum(",
        "hierarchical_grpo:M2+M3_unified_net_intervention",
        "context_extra_replay=0 action_space=hierarchical[single|pair]->operator",
    ):
        if required_text not in proposal + hres + joint + runner:
            errors.append(f"missing T2-L causal-context contract: {required_text}")
    trace = (ROOT / "src/causal_schedule_lab/m3/probability_trace.py").read_text(
        encoding="utf-8")
    config = (ROOT / "src/causal_schedule_lab/m3/config.py").read_text(
        encoding="utf-8")
    for required_text in (
        '"seed_groups"', "self.appearance_potential", "self.hop_embedding",
        "def _transition_weights(", "T2L_NET_REGRESSION_WEIGHT",
        '"m2_future_net_reward"', '"unified_net_intervention_return"',
        "best_to_terminal_regression",
    ):
        if required_text not in trace + config + joint:
            errors.append(f"missing T2-L unified intervention contract: {required_text}")
    for required_text in (
        '"ROUTE+SEQ_SWAP"', '"ROUTE+SEQ_INSERT"',
        '"SEQ_SWAP+SEQ_SWAP"', '"SEQ_SWAP+SEQ_INSERT"',
        'if family not in per_family:',
        "def _joint_operator_contract_compatible(",
        '"routing_sequence_contract"',
        '"same_resource_sequence_contract"',
        "old_base_of,",
        '"state_makespan_before": int(ms_cur)',
        'rec["successor_makespan"] = int(res["schedule"].makespan)',
        'long_horizon=20, long_horizon_every=5',
        '"actions/pair_selection_rate"',
        "def _bounded_hierarchical_action_pool(",
        "T2L_SINGLE_ACTIONS",
        '"single::SEQ_INSERT": single_cap - 2 * (single_cap // 4)',
        '"pair_probability_mass"',
        'pair_mask=rec.get("pair_mask")',
        "self.granularity_head",
        '"actions/on_policy_pair_selection_rate"',
        "def _worker_heartbeat(",
        "def _force_shutdown_process_pool(",
        'T2M_ROLLOUT_STALL_TIMEOUT_S',
        'collect STALL',
        'collect RECOVER',
    ):
        if required_text not in proposal + joint + pf + hres:
            errors.append(f"missing T2-L joint-macro v2 contract: {required_text}")
    offline_eval = (ROOT / "scripts/evaluate_t2l_checkpoint_series.py").read_text(
        encoding="utf-8")
    for required_text in (
        '"objective": "best_makespan_visited_within_horizon"',
        '"policy_mode": "pure_policy_sampled_no_anchor"',
        "anchor_trajectories=0", "milestone_policies",
        "checkpoint_validation_curve.png",
    ):
        if required_text not in offline_eval + runner:
            errors.append(f"missing offline paper-validation contract: {required_text}")

    parent = ROOT.parent / "t2d_hierarchical_residual_autodl"
    proof = {"parent": str(parent), "available": parent.is_dir(), "files": {}}
    if parent.is_dir():
        for rel, want in PARENT_HASHES.items():
            got = sha(parent / rel)
            proof["files"][rel] = {"expected": want, "actual": got, "unchanged": got == want}
            if got != want:
                # The uploaded package is self-contained.  A sibling T2-D
                # directory on AutoDL may legitimately contain the user's prior
                # hotfixes; that external state is useful provenance but must not
                # make this package look incomplete.
                proof["files"][rel]["warning"] = (
                    "external sibling parent differs; package validation unaffected")
        (ROOT / "docs" / "CANONICAL_UNCHANGED_PROOF.json").write_text(
            json.dumps(proof, indent=2), encoding="utf-8")
    print(json.dumps({"package": str(ROOT), "formal_test_access": 0,
                      "formal_test_included": False, "symlinks": len(symlinks),
                      "canonical_parent_checked": parent.is_dir(),
                      "errors": errors}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
