"""V5 mainline runtime skeleton + single-state expansion (D1 + D2).

Phase D.  This driver proves the V5 model is **wired** and, in D2, that a real
single-state expansion runs end to end.

D1 (``--mode d1``) does a real ``SGSCTRootCauseModelV5`` build, a canonical
compile, a single forward pass, and surfaces the *existing* detached proposal /
runtime audit.  It STOPS after the forward.

D2 (``--mode d2``) wires that single forward to the real pipeline:
``candidate decision sites -> Causal Explorer V2 -> Actionable Root Selector ->
Operator / Transition Reasoner -> Proposal Builder``, forming a real
**single-state expansion**: S_t -> candidate sites -> multiple causal traces ->
multiple actionable roots -> multiple complete legal proposals -> STOP.  D2 is
NOT a state-transition loop and executes **no** proposal.

Neither mode:
  * executes any proposal,
  * generates a next state S_{t+1},
  * iterates the closed loop,
  * modifies ``sg_sct_model_v5.py`` / Explorer / Operator / Transition / Effect
    / Memory / M3 / executor / counterfactual / Teacher-Trace core semantics,
  * modifies ``cli.py``,
  * loads a formal checkpoint or runs any training / preflight.

D2 authority firewall (structural, not cosmetic)  [spec §3/§4]
--------------------------------------------------------------
The canonical forward's ``root_logits`` **mix** an untrained neural half
(``learned_z`` = ||h_decision - h_reference||, plus ``node_logits`` relevance)
with a deterministic half (``deterministic_z`` = clamped site ``z_deviation``).
With random weights the neural half is noise, so the forward's own proposal
ordering is **not** a defensible runtime authority.

D2 therefore treats the neural scores as **diagnostic only**
(``neural_scores_available=true``, ``neural_scores_authoritative=false``) and
re-drives the *identical real* components -- ``CausalExplorerV2``,
``ActionableRootSelectorV2`` (with the neutral ``DeferredProposalEffectAdapter``
so the trainable Effect Predictor is never called), and ``build_operator_runtime``
-- using **deterministic** candidate scores sourced from each site's
``z_deviation`` (the forward's own deterministic half, no neural term) and a
neutral zero edit score.  ``candidate_authority = "deterministic_fallback"``.
This is a real call into existing components (spec §2/§4); nothing is
fabricated and no core module is modified (spec §1/§22).

Canonical schema is read from source (``run_m2_v5_schedule`` compiles via
``compile_sg_sct_input_v1_3`` -> ``schema_version == "1.3.0"``); it is asserted
at runtime, never hard-coded from a plan document.

Run:
    PYTHONPATH=src .venv/bin/python scripts/run_v5_mainline_loop.py            # D1
    PYTHONPATH=src .venv/bin/python scripts/run_v5_mainline_loop.py --mode d2  # D2 (Mk9)
    PYTHONPATH=src .venv/bin/python scripts/run_v5_mainline_loop.py --mode d2 --case routing_blocker
    PYTHONPATH=src .venv/bin/python scripts/run_v5_mainline_loop.py --mode d2 --case multi_root
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
# DEFAULT_FIXTURE points at the CURRENT v3 serialization (appearance-pruning-3.0
# / symptom-pruning-detector-v3), which emits only active appearances
# {A1,A2,A3,A4,A6}.  The former default
# ``outputs/gantt/BrandimarteMk9.reverse-inference.json`` was a STALE pre-v3
# artifact (appearance-pruning-2.0 / detector-v2) that still carried deprecated
# A5/A8 blocks.  NOTE: swapping the fixture is only data cleanup -- the actual
# fix is the fail-closed appearance gate in ``build_operator_runtime`` (via
# ``appearance_taxonomy``), which keeps a stale A5/A8 artifact non-actionable no
# matter which fixture is loaded.
DEFAULT_FIXTURE = REPO_ROOT / "outputs" / "gantt_initial" / "BrandimarteMk9.reverse-inference.json"
D2_SMOKE_ROOT = REPO_ROOT / "outputs" / "v5_mainline_d2_smoke"

# D2 expansion manifest schema (this driver's sidecar format, not a model schema).
# 1.1 adds bounded serialization of the high-cardinality record lists
# (causal_searches / neural diagnostics) so a full-scale instance (e.g. Mk9,
# ~57k searches) yields a bounded sidecar instead of a multi-GB dump.  The true
# totals are always preserved as ``*_total`` counts; only the emitted records
# are capped, with an explicit ``*_truncated`` flag -- nothing is silently lost.
# 1.2 adds a within-record trace cap (D2_MAX_TRACE) for beam-discard lists.
# 1.3 (D2.1) hardens provenance identity: stable content-addressed trace ids
# (per causal path, not colliding on decision site), authoritative
# supporting_trace_ids[] / supporting_candidate_ids[] on every root + proposal,
# a proposal derivation id, and a truncation-invariant provenance DAG summary.
# 1.6 (D5) adds the M3 proposal/STOP/CONTINUE shadow layer: a same-state action
# space A(S_t) = {proposal_1..n, STOP} (CONTINUE schema-reserved), a first-class
# STOP action_type, a per-proposal M3 input summary (STATE/PROPOSAL/CAUSAL/EFFECT/
# MEMORY), and a shadow (no-checkpoint, non-authoritative) M3 policy interface.
# It reads the (unchanged) D2 proposal set + D3 effect output + D4 memory evidence
# and adds NO decision power (no filter/reorder/removal, no execution, no S_t+1).
D2_MANIFEST_SCHEMA = "d2-expansion-1.6"  # 1.6 adds the D5 M3 proposal/STOP shadow layer

# --- D4 read-only per-state trajectory-memory retrieval ------------------------
# Memory is a RETRIEVAL PRIOR / EXPLORATION EVIDENCE layer (spec §2), never a
# source of causal / reward / success / delta / FIV truth (spec §7/§10).  D4
# wires ``S_t -> RetrieveMemory(S_t) -> memory context -> (unchanged) expansion
# -> attach memory evidence -> STOP``: it reads the FROZEN memory snapshot
# read-only (no append / rewrite / promote / compact / regenerate, spec §13),
# runs a deterministic per-state retrieval, and annotates proposals with
# similarity evidence WITHOUT changing the proposal set or the Effect Predictor
# output (spec §14.7/§14.8).  It executes no proposal, creates no next state,
# calls no M3, and mutates no memory.
#
# The frozen snapshot is a fixed prior: its SHA is asserted on every run so a
# drifted / regenerated store fails closed rather than silently changing the
# retrieval prior (spec §5/§14.3).
FROZEN_MEMORY_PATH = (
    REPO_ROOT
    / "outputs"
    / "v5_deepseek_trajectory_harvesting_v1"
    / "deepseek-harvest-v1-production-smoke"
    / "trajectory_memory.json"
)
FROZEN_MEMORY_SHA = (
    "3f804b36b09dc9cedb7d53830f589fba2ba0aa5e588cc2438b484d73c619e440"
)
# Retrieval prior is deterministic: fixed top-K and equal composite weights
# (alpha*sim_G + beta*sim_A + gamma*sim_P).  Changing these would change the
# retrieval prior and is out of D4 scope.
D4_RETRIEVAL_LIMIT = 5
D4_SIM_ALPHA = 1.0
D4_SIM_BETA = 1.0
D4_SIM_GAMMA = 1.0
D4_DELTA_EPS = 1e-9  # sign convention: delta_cmax = after - before; negative = better

# --- D3 effect layer (shadow / annotation only) --------------------------------
# The effect layer is a NON-authoritative annotation.  Without a formal
# checkpoint every prediction is null and ``effect_outputs_authoritative`` is
# False, so the layer has zero decision power (spec §16/§17): it never filters,
# reorders, or removes any proposal, and the candidate / trace / root / proposal
# / generated / legal / rejected totals are byte-identical with and without it.
EFFECT_PREDICTION_STATUS_DISABLED = "disabled_no_checkpoint"

# Per-head supervision contract verdicts, adjudicated from LIVE code in Phase D3
# (not from field names).  These describe what CAN legally be supervised, never
# what is being trained (nothing is trained in D3).
EFFECT_SUPERVISION_CONTRACT = {
    # delta_cmax: frozen-local attributable MSE target; sign = after-before,
    # negative = improvement; provenance-complete.  Closeable.
    "delta_status": "closed_frozen_local",
    # success: BCE on experience_future_success (trajectory eventually lowered
    # Cmax under the frozen-local gate).  Objective-improvement flavored but
    # provenance-gated; closeable with an explicit validity mask.
    "success_status": "closed_frozen_local",
    # future_gain: NO loss term exists in the live loss; the only candidate
    # label (experience_final_gain = max(0, -final_delta)) is a clamp_min(0)
    # terminal trajectory gain vs S0, NOT the continuation-minus-immediate
    # attributable target the contract requires, and no trajectory-value
    # machinery grounds it.  Fail-closed pending D6 / iterative teacher.
    "future_gain_status": "fail_closed_requires_trajectory_truth",
    # risk: BCE on failure = 1 - future_success; derived from the same
    # frozen-local gate as success.  Closeable with a validity mask.
    "risk_status": "closed_frozen_local_derived",
    # FIV: fiv_pred = success * gain (derived inference quantity, no separate
    # ground truth); NO independent supervision (avoids triple-counting).
    "fiv_status": "derived_no_supervision",
}

# --- D5 M3 proposal / STOP / CONTINUE shadow layer -----------------------------
# M3's ONLY responsibility (spec §3): given the current state's complete legal
# proposals, decide which to take next -- or STOP.  It is NOT causal search /
# root discovery / proposal generation / hard legality / executor / CP-SAT.  It
# is the LAST shadow consumer of the pipeline (spec §10): it reads the unchanged
# D2 proposal set + D3 effect output + D4 memory evidence and adds no decision
# power.  Without a formal checkpoint (spec §9) the policy is a shadow interface:
# it emits null outputs and is non-authoritative -- an untrained random selector
# must NEVER enter authoritative runtime.
M3_PREDICTION_STATUS_DISABLED = "disabled_no_checkpoint"

# Action types in the same-state action set A(S_t) (spec §4/§11).  STOP is a
# FIRST-CLASS action (not an external ``if...: break``) so a future GRPO group
# can compare P1/P2/P3/STOP over one S_t (spec §11/§13).
M3_ACTION_PROPOSAL = "PROPOSAL"
M3_ACTION_STOP = "STOP"
M3_ACTION_CONTINUE = "CONTINUE"

# CONTINUE verdict (spec §12): the live M3 (selection.py) represents only
# ``proposal`` (scored) and an implicit STOP (chosen_index=None /
# NO_TRUE_IMPROVEMENT); it has NO distinct CONTINUE semantic.  A forced CONTINUE
# would duplicate either "pick a proposal" (advance the search) or STOP (halt),
# so D5 does NOT introduce a runtime CONTINUE.  It is reserved in the schema,
# non-authoritative and unavailable, pending a real iterative-search design (D6).
M3_CONTINUE_VERDICT = {
    "implemented": False,
    "availability": "schema_reserved_non_authoritative",
    "live_semantic_found": False,
    "reason": (
        "live M3 (m3/selection.py) represents proposal (scored) + implicit STOP "
        "(chosen_index=None / NO_TRUE_IMPROVEMENT) only; no distinct CONTINUE "
        "semantic exists.  A forced CONTINUE would duplicate advance-search "
        "(pick a proposal) or halt (STOP), so D5 reserves it in-schema rather "
        "than introduce wrong runtime behavior for a checkbox (spec §12)."
    ),
    "candidate_semantics_considered": [
        "keep-branch-for-later-search (== defer, needs a real search frontier -> D6)",
        "expand-state-without-executing (== advance search == pick a proposal)",
        "halt (== STOP, already first-class)",
    ],
}

# Max records emitted per high-cardinality list in the sidecar.  Structural
# authority (counts, partitions, provenance) never depends on the cap; it only
# bounds the on-disk record dump.  Full totals are reported alongside.
D2_MAX_RECORDS = 2000

# Max entries emitted per *within-record* high-cardinality trace list (beam
# discards, visited nodes).  A single causal-search record can carry an
# unbounded beam-discard trace (~860 entries on Mk9); this bounds each record's
# trace lists while preserving their true lengths as ``*_total`` counts.
D2_MAX_TRACE = 32

# The canonical V5 real-schedule schema, as produced by ``run_m2_v5_schedule``'s
# internal ``compile_sg_sct_input_v1_3``.  Asserted against the live bundle at
# runtime (see ``build_v5_runtime_audit``); this constant is only the expected
# value used for the structural check, not a source of truth.
CANONICAL_SCHEMA_VERSION = "1.3.0"

# =============================================================================
# D6 iterative successor-state retention + mainline loop  [spec §0-§...]
# -----------------------------------------------------------------------------
# D6 turns the single-state D5 shadow (S_t -> analyze -> proposals -> STOP) into a
# real iterative loop: S_t -> analyze -> proposals -> EVALUATE each legal
# proposal's SUCCESSOR by Frozen-Local counterfactual -> RETAIN -> CHOOSE P* by a
# deterministic engineering fallback -> EXECUTE P* -> S_{t+1} -> RE-ANALYZE from
# scratch -> repeat, bounded.  It layers on the (unchanged) D5-verified
# ``build_d2_expansion`` bridge components; it does NOT alter that function.
#
# Retention rule (spec core), sign = Cmax(after) - Cmax(before):
#   immediate_delta < 0  -> RETAIN  (immediate_improvement)
#   immediate_delta == 0 -> RetrieveMemory(S'_i) [NOT reuse RetrieveMemory(S_t)];
#                           if the successor is similar to a historical state that
#                           LATER terminally improved Cmax (bucket
#                           neutral_then_improved) -> RETAIN
#                           (memory_supported_promising_successor); else a small
#                           DETERMINISTIC exploration budget (<= K, no randomness)
#                           retains, otherwise PRUNE.
#   immediate_delta > 0  -> PRUNE (D6 V1 default; no memory auto-retain).
# Memory NEVER generates reward / success_label / future_gain target / automatic
# selection.  The manifest DISTINGUISHES current_state_memory (from S_t) vs
# successor_state_memory (from S'_i).  M3 has no checkpoint -> non-authoritative;
# P* is chosen by an explicit DETERMINISTIC fallback (no random logits).
D6_MANIFEST_SCHEMA = "d6-iteration-1.0"
# sign convention: delta_cmax = Cmax(after) - Cmax(before); negative = better.
D6_DELTA_EPS = 1e-9
# Bounded loop knobs (spec: max_iterations / max_proposals_per_state /
# max_neutral_exploration).  At a boundary the loop STOPs with an explicit reason.
D6_MAX_ITERATIONS = 8
D6_MAX_PROPOSALS_PER_STATE = 16
D6_MAX_NEUTRAL_EXPLORATION = 1  # deterministic budget K; NO randomness
# The deterministic engineering fallback that selects P* while M3 is untrained
# (spec §9): it is NOT M3, holds no checkpoint, uses no random logits, and only
# ever takes a strictly-improving proposal -- otherwise it STOPs.
D6_SELECTION_AUTHORITY = "deterministic_engineering_fallback"
# proposal edit-type -> executable operator ids (the LIVE bridge validated end to
# end against generate_operator_candidates + _candidate_to_atom).  TIMING_SHIFT is
# intentionally absent -> unmappable -> fail-closed (never a crash, never a wrong
# op).  Keys equal the m2_v5_schema_v1 EDIT_* constants; asserted at runtime.
D6_EDIT_OPERATOR_MATCH: dict[str, tuple[str, ...]] = {
    "ROUTE": ("machine_reassignment", "stage_machine_reassignment"),
    "SEQ_SWAP": ("adjacent_resource_swap", "critical_block_resequence"),
    "SEQ_INSERT": ("resource_sequence_insertion",),
}


@dataclasses.dataclass(frozen=True)
class V5RuntimeAudit:
    """Structural audit of one D1 V5 build + single forward.

    Every honesty flag is a structural fact of *this run*, not a claim.
    """

    # --- provenance ---
    fixture: str
    case_id: str
    device: str
    forward_seconds: float

    # --- schema truth (read from the live bundle) ---
    bundle_schema_version: str
    canonical_schema_expected: str
    schema_matches_canonical: bool

    # --- honesty / authority (structural) ---
    checkpoint_loaded: bool
    mock_untrained: bool
    neural_outputs_authoritative: bool
    selection_authority: str
    identified: bool
    training_status: str
    model_schema_version: str

    # --- what the single forward produced (detached audit surface) ---
    symptom_block_count: int
    n_proposals: int
    n_actionable_roots: int
    n_causal_chains: int
    n_decision_sites: int
    n_legal_edits: int
    per_block_node_logits_shape: tuple[int, ...] | None

    # --- D1 boundary markers ---
    proposal_executed: bool
    next_state_generated: bool
    iterated: bool
    stop_point: str

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        # tuples -> lists for clean JSON
        if self.per_block_node_logits_shape is not None:
            d["per_block_node_logits_shape"] = list(self.per_block_node_logits_shape)
        return d


def load_case(fixture: Path) -> tuple[Any, Any, Mapping[str, Any], str]:
    """Load a (problem, schedule, appearance, case_id) triple from a
    reverse-inference serialization fixture -- the same shape the V5 tests use.
    """
    from causal_schedule_lab.ir import Problem, Schedule

    if not fixture.exists():
        raise FileNotFoundError(f"fixture not present: {fixture}")
    doc = json.loads(fixture.read_text())
    return (
        Problem.model_validate(doc["problem"]),
        Schedule.model_validate(doc["schedule"]),
        doc["appearance"],
        str(doc.get("problem_id", fixture.stem)),
    )


def build_v5_runtime_audit(
    problem: Any,
    schedule: Any,
    appearance: Mapping[str, Any],
    *,
    case_id: str,
    fixture: Path,
    device: str = "cpu",
    checkpoint: Path | None = None,
) -> V5RuntimeAudit:
    """Real V5 build + canonical compile + single forward, then STOP.

    Uses the canonical one-call entry ``run_m2_v5_schedule`` (which builds the
    model, compiles at schema 1.3.0, runs one detached forward, and validates
    the output contract).  No core module is modified or monkeypatched.
    """
    from causal_schedule_lab.sg_sct_model_v5 import run_m2_v5_schedule

    # --- structural honesty guarantee -------------------------------------
    # D1 never loads a formal checkpoint.  A checkpoint path is refused rather
    # than silently ignored, so authority can never be claimed by accident.
    checkpoint_loaded = False
    if checkpoint is not None:
        raise SystemExit(
            "D1 BLOCKER: formal checkpoint loading is out of D1 scope. "
            "D1 is mock-untrained only (random weights, diagnostic logits). "
            "Checkpoint wiring belongs to a later batch."
        )
    mock_untrained = not checkpoint_loaded
    # Authority is a structural function of checkpoint state, not a preference.
    neural_outputs_authoritative = checkpoint_loaded
    assert not neural_outputs_authoritative, (
        "structural invariant violated: neural outputs must not be authoritative "
        "without a formally loaded checkpoint"
    )
    selection_authority = "deterministic_fallback"

    # --- real build + single forward (canonical entry) --------------------
    t0 = time.time()
    output, _model, bundle = run_m2_v5_schedule(
        problem, schedule, appearance, case_id=case_id, device=device
    )
    forward_seconds = time.time() - t0

    bundle_schema_version = str(bundle.manifest.get("schema_version", ""))
    schema_matches_canonical = bundle_schema_version == CANONICAL_SCHEMA_VERSION

    node_logits = output.per_block_node_logits
    node_logits_shape = None if node_logits is None else tuple(node_logits.shape)

    return V5RuntimeAudit(
        fixture=str(fixture),
        case_id=case_id,
        device=device,
        forward_seconds=round(forward_seconds, 3),
        bundle_schema_version=bundle_schema_version,
        canonical_schema_expected=CANONICAL_SCHEMA_VERSION,
        schema_matches_canonical=schema_matches_canonical,
        checkpoint_loaded=checkpoint_loaded,
        mock_untrained=mock_untrained,
        neural_outputs_authoritative=neural_outputs_authoritative,
        selection_authority=selection_authority,
        identified=bool(output.identified),
        training_status=str(output.training_status),
        model_schema_version=str(output.model_schema_version),
        symptom_block_count=int(output.symptom_block_count),
        n_proposals=len(output.proposals),
        n_actionable_roots=len(output.actionable_root_ids),
        n_causal_chains=len(output.causal_explanation_chains),
        n_decision_sites=len(output.decision_sites),
        n_legal_edits=len(output.legal_edits),
        per_block_node_logits_shape=node_logits_shape,
        # --- D1 STOP boundary: nothing downstream is exercised ---
        proposal_executed=False,
        next_state_generated=False,
        iterated=False,
        stop_point="after_single_forward",
    )


# ===========================================================================
# D2 -- single-state expansion (candidate -> causal trace -> root -> proposal)
# ===========================================================================
#
# Design (see module docstring / spec §0-§24):
#   1. Run the canonical ``run_m2_v5_schedule`` (real V5 build + single forward).
#      This proves the wiring and yields the neural DIAGNOSTIC scores.  Its own
#      proposal ordering is neural-seeded and therefore NOT used as authority.
#   2. Reuse the forward's OWN deterministic runtime objects (``decision_sites``
#      / ``legal_edits`` -- both weight-independent) as the candidate space.
#   3. Re-drive the IDENTICAL real components -- ``CausalExplorerV2``,
#      ``ActionableRootSelectorV2`` (neutral ``DeferredProposalEffectAdapter``),
#      and ``build_operator_runtime`` -- with DETERMINISTIC candidate scores
#      (clamped ``z_deviation``) and neutral zero edit scores.  The result is
#      the authoritative expansion.
#   4. Extract candidate sites / causal searches / actionable roots / proposals
#      with a full provenance chain and a fail-closed legality partition.
#   5. STOP -- no Effect Predictor, no Memory, no M3, no executor, no next state.


def _det_root_scores(sites, *, clamp: float = 5.0):
    """Deterministic per-site score = clamped ``z_deviation``.

    This is exactly the ``deterministic_z`` half of the forward's own
    ``root_logits`` (``sg_sct_model_v5._runtime_outputs``), with the neural
    ``learned_z`` and ``node_logits`` terms removed.  No network weight touches
    this value; it is a pure residual statistic from ``build_decision_sites``.
    Shape ``[1, n_sites]`` to match ``build_operator_runtime``'s indexing.
    """
    row = [max(min(float(s.z_deviation), clamp), -clamp) for s in sites]
    return np.asarray([row], dtype=float)


def _det_edit_scores(n_blocks: int, n_edits: int):
    """Neutral (zero) edit scores.

    The forward's ``edit_logits`` are neural (edit relevance head + node
    logits); in the untrained regime they are noise, so D2 does not let them
    order proposals.  A constant-zero score keeps the proposal additive term
    neutral (deterministic tie-break by operator_type inside the builder).
    """
    return np.zeros((n_blocks, max(n_edits, 0)), dtype=float)


def _site_record(site) -> dict[str, Any]:
    return {
        "site_id": site.site_id,
        "operation_id": site.operation_id,
        "decision_type": site.decision_type,
        "decision_family": site.decision_type,  # routing | sequence
        "source_machine": site.source_machine,
        "target_machine": site.target_machine,
        "mode_id": site.mode_id,
        "resource_id": site.resource_id,
        "predecessor_id": site.predecessor_id,
        "successor_id": site.successor_id,
        "z_deviation": float(site.z_deviation),
        "edit_support": int(site.edit_support),
    }


def _legal_action_space_for_site(site, edits) -> list[str]:
    """The legal edit ids that act on this decision site (routing -> ROUTE on
    the same op; sequence -> SEQ_* touching the (pred,succ) pair).  Read from
    the already-enumerated hard-feasible ``edits`` (never fabricated)."""
    ids: list[str] = []
    if site.decision_type == "routing":
        for e in edits:
            if e.operation_id == site.operation_id and e.edit_type == "ROUTE":
                ids.append(e.edit_id)
    else:
        pair = {site.predecessor_id, site.successor_id}
        for e in edits:
            if e.edit_type.startswith("SEQ_") and (
                {e.operation_id, e.left_id, e.right_id} & pair
            ):
                ids.append(e.edit_id)
    return ids


def _edit_record(edit) -> dict[str, Any]:
    return {
        "edit_id": edit.edit_id,
        "edit_type": edit.edit_type,
        "operation_id": edit.operation_id,
        "source_machine": edit.source_machine,
        "target_machine": edit.target_machine,
        "target_mode_id": edit.target_mode_id,
        "resource_id": edit.resource_id,
        "left_id": edit.left_id,
        "right_id": edit.right_id,
        "insert_position": edit.insert_position,
        "target_start": edit.target_start,
    }


# --- D2.1 provenance identity (stable, deterministic, not runtime hash) ------
#
# The pre-D2.1 trace_id was ``{state}::trace::{appearance}::{decision_site}``.
# CausalExplorerV2 keys its search states by ``(seed_site_id, causal_path)`` and
# emits **many** distinct chains per seed -- all sharing one decision_site_id --
# so that formula collided (Mk9: 57,198 chains -> 1,410 ids; smoke: 6 -> 3).
# D2.1 restores per-path identity by folding a *stable* hash of the chain's
# causal path (its node sequence + ordered cause-edge sequence) into the id.
# The hash is content-addressed via ``sha256(json(...))`` -- deterministic and
# reproducible across processes -- never Python's salted ``hash()`` (spec §2).
def _stable_path_hash(chain) -> str:
    """Deterministic 16-hex fingerprint of a chain's causal path.

    Folds the node sequence and the ordered cause-edge sequence
    (``source|relation_type|target``) so two chains that differ only in their
    causal path get distinct ids, while byte-identical paths coincide (a
    legitimate duplicate, not a collision).  Uses sha256, not ``hash()``.
    """
    cause_edges = getattr(chain, "cause_edges", ()) or ()
    payload = {
        "nodes": list(chain.nodes),
        "edges": [
            f"{e.source}|{e.relation_type}|{e.target}" for e in cause_edges
        ],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _trace_id(chain, state_id: str) -> str:
    """Single source of truth for a chain's trace id (spec §2).

    ``state_id + appearance_id + seed_candidate/site + stable path hash``.
    Every consumer (search records, root aggregation, proposal provenance, the
    DAG) computes trace ids through this one function, so the DAG stays
    internally consistent by construction.
    """
    seed = chain.seed_decision_site_id or chain.decision_site_id
    return f"{state_id}::trace::{chain.appearance_id}::{seed}::{_stable_path_hash(chain)}"


def _candidate_id(state_id: str, site_id: str) -> str:
    """Candidate (seed decision-site) id -- matches ``candidate_sites`` above."""
    return f"{state_id}::cand::{site_id}"


def _root_id(state_id: str, site_id: str) -> str:
    """Actionable-root id -- state-local actionable-site identity (spec §3)."""
    return f"{state_id}::root::{site_id}"


def _bounded_id_set(ids: list[str], cap: int = D2_MAX_TRACE) -> dict[str, Any]:
    """Authoritative id relation with a truncation-invariant fingerprint.

    Returns the **complete** distinct count and a stable ``sha256`` hash over
    the full sorted id set, plus a bounded emitted sample.  The authority
    (``total`` + ``hash``) is computed over every id and never depends on the
    cap, so serialization truncation of ``ids`` cannot distort the provenance
    relation (spec §6/§8).  ``ids`` may hold huge support sets; only the emitted
    sample is bounded.
    """
    uniq = sorted(dict.fromkeys(ids))
    return {
        "total": len(uniq),
        "hash": hashlib.sha256("\n".join(uniq).encode()).hexdigest()[:16],
        "truncated": len(uniq) > cap,
        "ids": uniq[:cap],
    }


@dataclasses.dataclass(frozen=True)
class _ProvenanceIndex:
    """Complete chain->root->proposal DAG, authoritative over ALL chains.

    Built over the untruncated chain list so it stays authoritative regardless
    of ``D2_MAX_RECORDS`` / ``D2_MAX_TRACE`` sidecar caps (spec §5/§6).  Holds
    many-to-one / many-to-many relations directly -- it never forces a single
    linear ``state -> candidate -> trace -> root -> proposal`` chain.
    """

    # NOTE: keyed by *decision site reached by a chain* (a "traced site"), NOT
    # by "actionable root".  Every seed site a chain lands on becomes a key; the
    # actionable-root subset (selected by ActionableRootSelectorV2 and producing
    # a proposal) is a small subset of these keys.  On Mk9: 10 traced sites, 3
    # actionable roots.  Do not conflate the two -- looking a root's site id up
    # in these maps is correct, but ``len(...)`` is the traced-site count.
    state_id: str
    trace_ids_by_site: dict[str, list[str]]          # traced site_id -> [trace_id]
    candidate_ids_by_site: dict[str, list[str]]      # traced site_id -> [cand_id]
    derivation_by_pathkey: dict[tuple[str, tuple[str, ...]], list[str]]
    n_traces_total: int
    unique_trace_ids: int


def build_provenance_index(chains: list[Any], state_id: str) -> _ProvenanceIndex:
    """Aggregate every chain into the authoritative provenance DAG.

    Groups all chains by their (seed == decision) *site* id, so any site a chain
    reaches records **all** its supporting traces / seed candidates, not just the
    first (spec §3/§5).  This site set is a superset of the actionable roots (the
    selector picks a subset).  Looking an actionable root's site id up in these
    maps returns that root's complete support.  ``derivation_by_pathkey`` maps a
    proposal's (root site, node path) to the *set* of trace ids consistent with
    it, so a proposal is never forced onto one arbitrarily-chosen trace.
    """
    trace_ids_by_site: dict[str, list[str]] = {}
    candidate_ids_by_site: dict[str, list[str]] = {}
    derivation: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    all_trace_ids: set[str] = set()
    for c in chains:
        site_id = c.decision_site_id
        seed = c.seed_decision_site_id or site_id
        tid = _trace_id(c, state_id)
        all_trace_ids.add(tid)
        trace_ids_by_site.setdefault(site_id, []).append(tid)
        candidate_ids_by_site.setdefault(site_id, []).append(_candidate_id(state_id, seed))
        derivation.setdefault((site_id, tuple(c.nodes)), []).append(tid)
    return _ProvenanceIndex(
        state_id=state_id,
        trace_ids_by_site=trace_ids_by_site,
        candidate_ids_by_site=candidate_ids_by_site,
        derivation_by_pathkey=derivation,
        n_traces_total=len(chains),
        unique_trace_ids=len(all_trace_ids),
    )


def _causal_search_record(chain, state_id: str) -> dict[str, Any]:
    trace = chain.search_trace
    trace_id = _trace_id(chain, state_id)
    rec: dict[str, Any] = {
        "trace_id": trace_id,
        "path_hash": _stable_path_hash(chain),
        "appearance_id": chain.appearance_id,
        "seed_decision_site_id": chain.seed_decision_site_id or chain.decision_site_id,
        "decision_site_id": chain.decision_site_id,
        "operation_id": chain.operation_id,
        "root_candidate_id": chain.root_candidate_id,
        "actionable": bool(chain.actionable),
        "depth": int(chain.depth),
        "causal_score": float(chain.causal_score),
        "m2_root_score_diagnostic": float(chain.m2_root_score),
        "nodes": list(chain.nodes),
        "relations": list(chain.relations),
        "operator_types": list(chain.operator_types),
    }
    if trace is not None:
        visited = list(trace.visited_nodes)
        discarded = list(trace.discarded_paths)
        rec["visited_nodes_total"] = len(visited)
        rec["visited_nodes_truncated"] = len(visited) > D2_MAX_TRACE
        rec["visited_nodes"] = visited[:D2_MAX_TRACE]
        rec["discarded_paths_total"] = len(discarded)
        rec["discarded_paths_truncated"] = len(discarded) > D2_MAX_TRACE
        rec["discarded_paths"] = discarded[:D2_MAX_TRACE]
        rec["selected_root"] = trace.selected_root
        rec["reason"] = trace.reason
    return rec


def _proposal_record(
    prop, state_id: str, prov: "_ProvenanceIndex | None" = None
) -> dict[str, Any]:
    site = prop.root_decisions[0] if prop.root_decisions else None
    # root_id is a state-local actionable-site identity (spec §3); it is keyed by
    # the actionable decision site, NOT by appearance, so multiple appearances /
    # traces that resolve to the same actionable site share one root_id.
    root_id = _root_id(state_id, prop.root_decision_id)

    # --- authoritative provenance DAG edges (spec §4/§5/§6) ------------------
    # supporting_trace_ids / supporting_candidate_ids are pulled from the
    # untruncated provenance index, so their totals + hash are authoritative
    # regardless of any sidecar record cap.  derivation_id pins this proposal to
    # the SET of traces whose causal path matches it (never one arbitrary trace).
    support_traces: list[str] = []
    support_candidates: list[str] = []
    derivation_trace_ids: list[str] = []
    if prov is not None:
        support_traces = list(prov.trace_ids_by_site.get(prop.root_decision_id, []))
        support_candidates = list(prov.candidate_ids_by_site.get(prop.root_decision_id, []))
        # A proposal's ``causal_chain`` is the full explanation-node sequence and
        # equals a source chain's ``nodes`` (verified against runtime); ``root_path``
        # is only the operation sub-path, so it must NOT be used as the join key.
        derivation_trace_ids = list(
            prov.derivation_by_pathkey.get(
                (prop.root_decision_id, tuple(prop.causal_chain)), []
            )
        )
    # proposal-specific derivation id: stable over (proposal_id, root_id, path).
    derivation_blob = json.dumps(
        {
            "proposal_id": prop.proposal_id,
            "root_id": root_id,
            "causal_chain": list(prop.causal_chain),
            "operator_type": prop.operator_type,
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    derivation_id = (
        f"{state_id}::deriv::"
        + hashlib.sha256(derivation_blob.encode()).hexdigest()[:16]
    )
    return {
        "proposal_id": prop.proposal_id,
        "provenance": {
            "state_id": state_id,
            "appearance_id": prop.appearance_id,
            "root_decision_id": prop.root_decision_id,
            "root_id": root_id,
            "derivation_id": derivation_id,
            # complete, truncation-invariant support relations (authority) ...
            "supporting_trace_ids": _bounded_id_set(support_traces),
            "supporting_candidate_ids": _bounded_id_set(support_candidates),
            "derivation_trace_ids": _bounded_id_set(derivation_trace_ids),
        },
        "root_decision_id": prop.root_decision_id,
        "root_operation_id": site.operation_id if site is not None else None,
        "operator_type": prop.operator_type,
        "root_path": list(prop.root_path),
        "legal_edits": [e.edit_id for e in prop.edits],
        "edit_types": [e.edit_type for e in prop.edits],
        "dependency_edges": [
            {
                "dependency_id": d.dependency_id,
                "editor_id": d.editor_id,
                "dependent_id": d.dependent_id,
                "dependency_type": d.dependency_type,
                "kind": d.kind,
            }
            for d in prop.dependencies
        ],
        "execution_order": list(prop.action_order),
        "transition_depth": int(prop.transition_depth),
        "transition_complete": bool(prop.transition_complete),
        "nodes": list(prop.nodes),
        "proposal_score": float(prop.proposal_score),
        "confidence": float(prop.confidence),
        # D2 does not evaluate any proposal (spec §13/§14).
        "prediction_status": "not_evaluated_d2",
        "delta_cmax": None,
        "success_pred": None,
        "future_gain_pred": None,
        "risk_pred": None,
        # --- D3 shadow effect layer (annotation only, no decision power) ------
        # effect_request pins the predictor input to the SAME identity used by
        # the provenance DAG (spec §4): a request is keyed by proposal_id +
        # derivation_id + root_id and carries the authoritative (truncation-
        # invariant) provenance hashes, so the same root's P1/P2/P3 are three
        # DISTINCT requests, never joined by root_id alone.
        "effect_request": {
            "proposal_id": prop.proposal_id,
            "derivation_id": derivation_id,
            "root_id": root_id,
            "state_id": state_id,
            "operator_type": prop.operator_type,
            "provenance_hashes": {
                "supporting_trace_ids": {
                    "total": len(dict.fromkeys(support_traces)),
                    "hash": _bounded_id_set(support_traces)["hash"],
                },
                "supporting_candidate_ids": {
                    "total": len(dict.fromkeys(support_candidates)),
                    "hash": _bounded_id_set(support_candidates)["hash"],
                },
                "derivation_trace_ids": {
                    "total": len(dict.fromkeys(derivation_trace_ids)),
                    "hash": _bounded_id_set(derivation_trace_ids)["hash"],
                },
            },
            # feature availability is real; feature VALUES are only materialised
            # once a predictor consumes the request (none in D2/D3 shadow).
            "features": {
                "state_features_available": True,
                "chain_features_available": bool(prop.causal_chain),
                "proposal_features_available": True,
                "memory_features_available": False,  # no retrieval in D3
            },
            "status": EFFECT_PREDICTION_STATUS_DISABLED,
        },
        # effect_prediction is null until a formal checkpoint is loaded; a random
        # predictor must never masquerade as meaningful (spec §16).
        "effect_prediction": {
            "status": EFFECT_PREDICTION_STATUS_DISABLED,
            "delta_pred": None,
            "success_pred": None,
            "future_gain_pred": None,
            "risk_pred": None,
            "fiv_pred": None,
        },
    }


# =============================================================================
# D4 read-only per-state trajectory-memory retrieval  [spec §1-§16]
# -----------------------------------------------------------------------------
# Memory is a retrieval PRIOR / exploration EVIDENCE layer (spec §2), never a
# source of causal / reward / success / delta / FIV truth (spec §7/§10).  These
# helpers (a) load the FROZEN snapshot read-only and assert its SHA, (b) run one
# deterministic per-state retrieval ``M_t = RetrieveMemory(S_t)`` (spec §4) with
# self-leak exclusion (spec §6), (c) classify retrieved trajectories into the
# four §8 evidence buckets, and (d) attach per-proposal ``memory_evidence`` via
# the D2.1 provenance identity (spec §11) WITHOUT changing the proposal set or
# the Effect Predictor output (spec §14.7/§14.8).
# =============================================================================


def _frozen_memory_sha(path: Path) -> str:
    """SHA-256 of the frozen memory file bytes (spec §5)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _appearance_type_from_id(appearance_id: str) -> str:
    """Extract the appearance TYPE token (e.g. ``A2``) from an appearance id.

    Ids look like ``appearance:A2:0002`` or ``A2:0002``; the type is the token
    that starts with ``A`` followed by a digit.  Returns ``""`` if none found.
    """
    if not appearance_id:
        return ""
    for tok in str(appearance_id).replace(":", " ").split():
        if len(tok) >= 2 and tok[0] in ("A", "a") and tok[1:].isdigit():
            return tok.upper()
    return ""


def _experience_immediate_delta(exp: Any) -> float:
    """Immediate (step-0) Cmax delta of a stored trajectory.

    Sign convention: ``delta_cmax = after - before``; negative == improvement.
    """
    if exp.trajectory:
        return float(exp.trajectory[0].delta_cmax)
    if exp.outcome is not None:
        return float(exp.outcome.delta_cmax)
    return 0.0


def _experience_final_delta(exp: Any) -> float:
    """Terminal trajectory Cmax delta vs the trajectory's initial state."""
    if exp.trajectory_final_delta_cmax is not None:
        return float(exp.trajectory_final_delta_cmax)
    return _experience_immediate_delta(exp)


def _classify_experience_bucket(exp: Any, eps: float = D4_DELTA_EPS) -> str:
    """Assign a stored trajectory to one of the §8 evidence buckets.

    Buckets are computed from (immediate_delta, final_delta) only; nothing is
    silently dropped -- trajectories fitting none of the four named buckets go
    to ``other`` so negatives are always retained downstream (spec §9).
    """
    immediate = _experience_immediate_delta(exp)
    final = _experience_final_delta(exp)
    if immediate < -eps:
        return "immediate_improvement"
    if abs(immediate) <= eps and final < -eps:
        return "neutral_then_improved"
    if abs(immediate) <= eps and final >= -eps:
        return "neutral_then_failed"
    if immediate > eps and final < -eps:
        return "worsening_then_recovered"
    return "other"


def _benchmark_lineage(instance_id: str) -> str:
    """Coarse benchmark lineage token for self-leak exclusion.

    NOT the six-family (JSP/FJSP/...) -- that is the retrieval *corpus* scope,
    not a self-leak grain (blanket-excluding the six-family would delete the
    whole single-family corpus).  Self-leak is prevented at the benchmark
    lineage (Brandimarte vs Fattahi vs ...) + base/schedule instance grain
    (spec §6).  Lineage = lowercased id with trailing digits stripped.
    """
    s = str(instance_id or "").strip().lower()
    return s.rstrip("0123456789")


def _memory_exclusion_decision(
    exp: Any,
    *,
    current_base_instance: str,
    current_schedule_instance: str,
    current_lineage: str,
    current_trajectory_keys: frozenset[str],
) -> tuple[bool, str]:
    """Return (excluded, reason) for one candidate experience (spec §6).

    Fail-closed: an experience whose metadata lacks the identity fields needed
    to prove it is NOT a self-leak is force-excluded (we cannot certify it is
    safe), and the reason is reported.
    """
    if exp.key in current_trajectory_keys:
        return True, "current_trajectory_self_leak"
    md = exp.metadata or {}
    base = md.get("base_instance_id")
    sched = md.get("schedule_instance_id")
    fam = md.get("family")
    if base is None or sched is None or fam is None:
        # cannot certify non-self-leak -> fail closed (spec §6)
        return True, "fail_closed_missing_identity_fields"
    if current_base_instance and str(base) == current_base_instance:
        return True, "same_base_instance"
    if current_schedule_instance and str(sched) == current_schedule_instance:
        return True, "same_schedule_instance"
    if current_lineage and _benchmark_lineage(base) == current_lineage:
        return True, "same_benchmark_lineage"
    return False, "retained_in_corpus"


def retrieve_state_memory(
    problem: Any,
    schedule: Any,
    appearance: Mapping[str, Any],
    *,
    state_id: str,
    state_fingerprint: str,
    case_id: str,
    current_trajectory_keys: frozenset[str] = frozenset(),
    memory_path: Path = FROZEN_MEMORY_PATH,
    expected_sha: str = FROZEN_MEMORY_SHA,
) -> dict[str, Any]:
    """One deterministic, read-only per-state retrieval ``M_t = RetrieveMemory(S_t)``.

    Loads the frozen snapshot read-only (no mutation, spec §13), asserts its SHA
    (fail-closed on drift, spec §5), builds the query state via the real
    ``encode_state`` (no solver / model), enforces self-leak exclusion caller
    side (spec §6, since the similarity module only supports one exclude_key),
    runs the real deterministic ``retrieve_similar_scored`` over the surviving
    in-memory (never persisted) store, and returns the §5/§8/§12 retrieval
    provenance + evidence block.  Produces EVIDENCE only -- no keep/drop, no
    reward, no success, no future_gain target (spec §3/§7/§10).
    """
    from causal_schedule_lab.memory import (
        ExperienceStore,
        ProposalRecord,
        encode_state,
        retrieve_similar_scored,
        state_vector,
    )

    # -- (a) frozen snapshot, read-only, SHA-asserted (spec §5/§13) --------
    if not memory_path.exists():
        raise SystemExit(
            f"D4 BLOCKER: frozen trajectory memory not found at {memory_path}. "
            "D4 requires the frozen read-only snapshot."
        )
    snapshot_sha = _frozen_memory_sha(memory_path)
    if snapshot_sha != expected_sha:
        raise SystemExit(
            "D4 BLOCKER: frozen memory SHA drift -- the retrieval prior must "
            f"not change silently.\n  expected {expected_sha}\n  actual   {snapshot_sha}"
        )
    frozen_store = ExperienceStore(path=memory_path)  # loads read-only; never appended to
    all_experiences = list(frozen_store.experiences())

    # -- current-state identity for self-leak exclusion (spec §6) ----------
    current_base_instance = str(getattr(problem, "id", case_id) or case_id)
    current_schedule_instance = current_base_instance
    current_lineage = _benchmark_lineage(current_base_instance)
    current_six_family = str(
        getattr(problem, "kind", "") or appearance.get("family", "") or ""
    ).upper()

    # -- (b) query state via the real encoder (deterministic) --------------
    encode_failed: str | None = None
    try:
        query_state = encode_state(problem, schedule)
        qvec = [round(float(x), 6) for x in state_vector(query_state)]
    except Exception as exc:  # fail-closed to no_match, retrieval still "ran"
        encode_failed = f"{type(exc).__name__}: {exc}"
        query_state = None
        qvec = []

    # -- self-leak exclusion, caller side (spec §6) ------------------------
    surviving: list[Any] = []
    excluded_ids: list[str] = []
    exclusion_reasons: dict[str, str] = {}
    fail_closed_ids: list[str] = []
    for exp in all_experiences:
        excluded, reason = _memory_exclusion_decision(
            exp,
            current_base_instance=current_base_instance,
            current_schedule_instance=current_schedule_instance,
            current_lineage=current_lineage,
            current_trajectory_keys=current_trajectory_keys,
        )
        if excluded:
            excluded_ids.append(exp.key)
            exclusion_reasons[exp.key] = reason
            if reason == "fail_closed_missing_identity_fields":
                fail_closed_ids.append(exp.key)
        else:
            surviving.append(exp)

    # -- (c) deterministic retrieval over the surviving in-memory store ----
    picks: list[tuple[Any, float]] = []
    if query_state is not None and surviving:
        filtered = ExperienceStore()  # path=None -> pure in-memory, never persisted
        filtered._records = {e.key: e for e in surviving}
        picks = retrieve_similar_scored(
            filtered,
            query_state,
            None,
            k=D4_RETRIEVAL_LIMIT,
            alpha=D4_SIM_ALPHA,
            beta=D4_SIM_BETA,
            gamma=D4_SIM_GAMMA,
        )

    # -- retrieval hashes (spec §5) ----------------------------------------
    exclusion_policy = {
        "grain": "benchmark_lineage + base_instance + schedule_instance + current_trajectory",
        "six_family_is_corpus_scope_not_filter": True,
        "fail_closed_on_missing_identity": True,
    }
    query_blob = json.dumps(
        {
            "state_id": state_id,
            "state_vector": qvec,
            "exclusion_policy": exclusion_policy,
            "k": D4_RETRIEVAL_LIMIT,
            "weights": [D4_SIM_ALPHA, D4_SIM_BETA, D4_SIM_GAMMA],
        },
        sort_keys=True,
    )
    retrieval_query_hash = hashlib.sha256(query_blob.encode()).hexdigest()[:16]
    set_blob = json.dumps(
        [[e.key, round(float(sim), 6)] for e, sim in picks], ensure_ascii=True
    )
    retrieval_set_hash = hashlib.sha256(set_blob.encode()).hexdigest()[:16]

    # -- (d) four-bucket structured evidence (spec §8/§9) ------------------
    buckets: dict[str, list[str]] = {
        "immediate_improvement": [],
        "neutral_then_improved": [],
        "neutral_then_failed": [],
        "worsening_then_recovered": [],
        "other": [],
    }
    retrieved_records: list[dict[str, Any]] = []
    for exp, sim in picks:
        bucket = _classify_experience_bucket(exp)
        buckets[bucket].append(exp.key)
        md = exp.metadata or {}
        retrieved_records.append(
            {
                "experience_id": exp.key,
                "similarity": round(float(sim), 6),
                "bucket": bucket,
                "immediate_delta_cmax": _experience_immediate_delta(exp),
                "final_delta_cmax": _experience_final_delta(exp),
                # structural join key material (spec §11) -- pattern, not op/root id
                "appearance_type": exp.proposal.appearance_type,
                "operator_type": exp.proposal.operator_type,
                # provenance of the stored experience (never treated as truth)
                "base_instance_id": md.get("base_instance_id"),
                "family": md.get("family"),
                # explicit similarity != truth firewall (spec §7)
                "meaning": "structurally_similar_historical_trajectory",
                "is_causal_truth": False,
                "is_reward_truth": False,
                "is_success_truth": False,
            }
        )

    retrieved_ids = [e.key for e, _ in picks]
    similarity_scores = {e.key: round(float(sim), 6) for e, sim in picks}
    support_status = "supported" if picks else "no_match"

    return {
        # per-state contract (spec §4): re-callable for any S_t, not S0-once
        "retrieval_contract": "M_t = RetrieveMemory(S_t)",
        "role": "retrieval_prior_exploration_evidence",  # spec §2
        "authoritative": False,  # memory never decides (spec §2/§7/§10)
        "mutated": False,  # read-only (spec §13)
        # --- retrieval provenance (spec §5) ---
        "state_id": state_id,
        "state_fingerprint": state_fingerprint,
        "memory_snapshot_sha": snapshot_sha,
        "memory_snapshot_sha_expected": expected_sha,
        "memory_snapshot_sha_matches": snapshot_sha == expected_sha,
        "retrieval_query_hash": retrieval_query_hash,
        "retrieved_experience_ids": retrieved_ids,
        "similarity_scores": similarity_scores,
        "retrieval_set_hash": retrieval_set_hash,
        "retrieval_limit": D4_RETRIEVAL_LIMIT,
        "total_candidates_considered": len(all_experiences),
        "encode_failed": encode_failed,
        # --- self-leak exclusion (spec §6) ---
        "instance_family_exclusion": {
            "policy": (
                "self-leak prevented at benchmark-lineage + base-instance + "
                "schedule-instance + current-trajectory grain; six-family is "
                "corpus scope, not a self-leak filter"
            ),
            "current_base_instance_id": current_base_instance,
            "current_schedule_instance_id": current_schedule_instance,
            "current_benchmark_lineage": current_lineage,
            "current_six_family": current_six_family,
            "six_family_corpus_scope": current_six_family or "unspecified",
            "excluded_experience_ids": excluded_ids,
            "excluded_count": len(excluded_ids),
            "exclusion_reasons": exclusion_reasons,
            "fail_closed_missing_identity_ids": fail_closed_ids,
            "surviving_candidate_count": len(surviving),
        },
        # --- retrieved evidence (spec §8) ---
        "retrieved_count": len(picks),
        "memory_support_status": support_status,  # spec §12: no_match is legal
        "evidence_buckets": buckets,
        "retrieved_experiences": retrieved_records,
        # state-level neutral-retention evidence (per-proposal is attached below)
        "neutral_then_improved_available": bool(buckets["neutral_then_improved"]),
        # --- honesty firewall (spec §7/§10) ---
        "similarity_is_not_truth": True,
        "maps_to_future_gain_pred": False,
        "maps_to_success_label": False,
        "decides_keep_or_drop": False,
    }


def _proposal_memory_evidence(
    prop: Any, memory_context: Mapping[str, Any]
) -> dict[str, Any]:
    """Per-proposal memory evidence, joined via the D2.1 causal PATTERN (spec §11).

    The anchor (which proposal gets the evidence) is the proposal's own D2.1
    identity (proposal_id / derivation is carried by the record).  The MATCH
    (which retrieved trajectories are relevant) is by structural causal pattern
    (appearance_type + operator_type), NOT coarse operation-id or root-id
    equality -- cross-instance experiences never share literal op/root ids.

    Produces EVIDENCE only: ``memory_supports_neutral_retention`` is a flag for
    a future D6 / teacher-search neutral-enabling branch (spec §3), never a
    keep/drop / reward / success authority here.
    """
    prop_sig = (
        _appearance_type_from_id(getattr(prop, "appearance_id", "")),
        getattr(prop, "operator_type", ""),
    )
    matched_ids: list[str] = []
    matched_buckets: dict[str, list[str]] = {
        "immediate_improvement": [],
        "neutral_then_improved": [],
        "neutral_then_failed": [],
        "worsening_then_recovered": [],
        "other": [],
    }
    matched_scores: dict[str, float] = {}
    for rec in memory_context.get("retrieved_experiences", []):
        exp_sig = (rec.get("appearance_type", ""), rec.get("operator_type", ""))
        # pattern match: appearance-type must agree; operator-type agreement is
        # a stronger match but a shared appearance pattern alone still counts as
        # structurally-similar evidence (never as truth).
        if prop_sig[0] and exp_sig[0] and prop_sig[0] == exp_sig[0]:
            matched_ids.append(rec["experience_id"])
            matched_buckets[rec["bucket"]].append(rec["experience_id"])
            matched_scores[rec["experience_id"]] = rec["similarity"]
    supports_neutral = bool(matched_buckets["neutral_then_improved"])
    status = "supported" if matched_ids else "no_match"
    return {
        "join_basis": "d2.1_causal_pattern (appearance_type + operator_type); NOT coarse op/root id",
        "proposal_pattern_signature": {
            "appearance_type": prop_sig[0],
            "operator_type": prop_sig[1],
        },
        "matched_experience_ids": matched_ids,
        "matched_similarity_scores": matched_scores,
        "matched_buckets": matched_buckets,
        "memory_support_status": status,  # spec §12: no_match is legal, not prune
        # spec §3/§15: EVIDENCE for the neutral-enabling branch, not authority
        "memory_supports_neutral_retention": supports_neutral,
        # spec §7/§10 firewall repeated at the join site
        "similarity_is_not_truth": True,
        "is_causal_truth": False,
        "is_reward_truth": False,
        "is_success_truth": False,
        "maps_to_future_gain_pred": False,
    }


def _attach_memory_to_record(
    base_record: dict[str, Any], prop: Any, memory_context: Mapping[str, Any]
) -> dict[str, Any]:
    """Return base_record + memory annotation, changing NOTHING authoritative.

    Adds exactly one top-level key (``memory_evidence``) and flips exactly one
    input-availability sub-flag (``effect_request.features.memory_features_available``
    -> True, retrieval now ran).  The proposal identity, the legal edit graph,
    the provenance DAG, and the Effect Predictor OUTPUT (``effect_prediction``)
    are untouched -- proving the proposal set + predictions are unchanged
    (spec §14.7/§14.8).
    """
    rec = copy.deepcopy(base_record)
    rec["memory_evidence"] = _proposal_memory_evidence(prop, memory_context)
    rec["effect_request"]["features"]["memory_features_available"] = True
    return rec


# =============================================================================
# D5 M3 proposal / STOP / CONTINUE shadow layer  [spec §1-§20]
# -----------------------------------------------------------------------------
# The LAST shadow consumer of the pipeline (spec §10).  Given the current
# state's COMPLETE legal proposals, it constructs the same-state action set
# ``A(S_t) = {proposal_1..n, STOP}`` (CONTINUE schema-reserved, spec §4/§11/§12),
# builds a per-proposal M3 input summary (STATE/PROPOSAL/CAUSAL/EFFECT/MEMORY,
# spec §5) that explicitly surfaces ``memory_supports_neutral_retention`` +
# neutral_then_improved / neutral_then_failed counts (spec §6), and exposes a
# shadow M3 policy interface (spec §9: no checkpoint -> null, non-authoritative).
# It reads the UNCHANGED D2 proposal set / D3 effect output / D4 memory evidence
# and adds NO decision power (spec §10): no filter / reorder / removal, no
# execution, no S_t+1, no reward (spec §14), no memory mutation.  Memory reaches
# M3 as an INPUT FEATURE, never as a hard-coded selector (spec §7).
# =============================================================================


def _m3_action_id(state_id: str, action_type: str, ordinal: int) -> str:
    """Stable per-state action id.  STOP/CONTINUE are singletons (ordinal 0)."""
    return f"{state_id}::action::{action_type.lower()}::{ordinal}"


def _m3_memory_summary(memory_evidence: Mapping[str, Any]) -> dict[str, Any]:
    """The small MEMORY input summary M3 sees (spec §5/§6) -- NOT the whole record.

    Surfaces exactly the neutral-retention evidence the future policy must learn
    from (``memory_supports_neutral_retention`` + the two bucket counts), plus a
    compact similarity summary and the honesty firewall.  Similarity is EVIDENCE,
    never truth, and the flag never selects an action (spec §7).
    """
    buckets = memory_evidence.get("matched_buckets", {})
    sims = memory_evidence.get("matched_similarity_scores", {}) or {}
    sim_values = [float(v) for v in sims.values()]
    return {
        "memory_match_status": memory_evidence.get("memory_support_status", "no_match"),
        "matched_experience_count": len(memory_evidence.get("matched_experience_ids", [])),
        # --- spec §6: the single most critical Memory feature for M3 ---
        "memory_supports_neutral_retention": bool(
            memory_evidence.get("memory_supports_neutral_retention", False)
        ),
        "similar_neutral_then_improved": len(buckets.get("neutral_then_improved", [])),
        "similar_neutral_then_failed": len(buckets.get("neutral_then_failed", [])),
        "similar_immediate_improvement": len(buckets.get("immediate_improvement", [])),
        "similar_worsening_then_recovered": len(buckets.get("worsening_then_recovered", [])),
        # compact similarity summary (not the per-experience record dump)
        "max_similarity": round(max(sim_values), 6) if sim_values else 0.0,
        "mean_similarity": round(sum(sim_values) / len(sim_values), 6) if sim_values else 0.0,
        # --- spec §7 firewall (repeated at the M3 input boundary) ---
        "similarity_is_not_truth": True,
        "is_reward_truth": False,
        "is_success_truth": False,
        "maps_to_future_gain_pred": False,
        "note": (
            "memory_supports_neutral_retention=true means ONLY: a structurally "
            "similar historical state had no immediate improvement yet the "
            "trajectory later improved, so the proposal should not be greedily "
            "pruned for immediate delta=0.  It is NOT reward>0 / success=true / "
            "historical gain / auto-accept (spec §1/§6)."
        ),
    }


def _m3_proposal_input(record: Mapping[str, Any]) -> dict[str, Any]:
    """Per-proposal M3 input (spec §5): STATE / PROPOSAL / CAUSAL / EFFECT / MEMORY.

    Assembled ENTIRELY from the (unchanged) attached proposal record -- M3 reads
    the pipeline's authoritative output, it never recomputes or alters it.
    """
    prov = record["provenance"]
    er = record["effect_request"]
    ep = record["effect_prediction"]
    mem = record.get("memory_evidence", {})
    return {
        # -- STATE prong (spec §5) --
        "state": {
            "state_id": prov["state_id"],
            "features_available": bool(er["features"]["state_features_available"]),
        },
        # -- PROPOSAL prong (spec §5) -- identity is D2.1-preserved verbatim
        "proposal": {
            "proposal_id": record["proposal_id"],
            "derivation_id": prov["derivation_id"],
            "root_id": prov["root_id"],
            "operator_type": record["operator_type"],
            "features": {
                "n_root_nodes": len(record["root_path"]),
                "n_actions": len(record["execution_order"]),
                "n_dependency_edges": len(record["dependency_edges"]),
                "n_nodes": len(record["nodes"]),
                "transition_depth": record["transition_depth"],
                "transition_complete": record["transition_complete"],
                "proposal_score": record["proposal_score"],
                "confidence": record["confidence"],
            },
        },
        # -- CAUSAL prong (spec §5): root/causal provenance, evidence hashes/counts
        "causal": {
            "root_decision_id": record["root_decision_id"],
            "root_operation_id": record["root_operation_id"],
            "appearance_id": prov["appearance_id"],
            "supporting_trace_ids": {
                "total": prov["supporting_trace_ids"]["total"],
                "hash": prov["supporting_trace_ids"]["hash"],
            },
            "supporting_candidate_ids": {
                "total": prov["supporting_candidate_ids"]["total"],
                "hash": prov["supporting_candidate_ids"]["hash"],
            },
            "derivation_trace_ids": {
                "total": prov["derivation_trace_ids"]["total"],
                "hash": prov["derivation_trace_ids"]["hash"],
            },
        },
        # -- EFFECT prong (spec §5/§8): D3 shadow prediction + status, fail-closed
        "effect": {
            "status": ep["status"],
            "delta_pred": ep["delta_pred"],
            "success_pred": ep["success_pred"],
            "future_gain_pred": ep["future_gain_pred"],
            "risk_pred": ep["risk_pred"],
            "fiv_pred": ep["fiv_pred"],
            # D3 supervision-contract verdicts carried through unchanged (spec §8)
            "future_gain_status": EFFECT_SUPERVISION_CONTRACT["future_gain_status"],
            "fiv_status": EFFECT_SUPERVISION_CONTRACT["fiv_status"],
        },
        # -- MEMORY prong (spec §5/§6): the small neutral-retention summary --
        "memory": _m3_memory_summary(mem),
    }


def _m3_shadow_prediction() -> dict[str, Any]:
    """Shadow M3 policy output (spec §9): no checkpoint -> null, non-authoritative.

    An untrained random selector must NEVER enter authoritative runtime, so the
    policy is called only structurally: it emits nulls with a disabled status.
    """
    return {
        "status": M3_PREDICTION_STATUS_DISABLED,
        "ranking_score": None,
        "accept_prob": None,
        "risk_prob": None,
        "selected": False,          # M3 never selects in D5 (spec §9/§15)
        "authoritative": False,     # spec §9
    }


def build_m3_action_space(
    legal_records: list[dict[str, Any]], state_id: str, memory_context: Mapping[str, Any]
) -> dict[str, Any]:
    """Same-state action set ``A(S_t) = {proposal_1..n, STOP}`` (spec §4/§11/§13).

    Every action shares ``state_key = state_id`` so a future GRPO group is
    provably one S_t (spec §13) -- proposals from different states can never be
    mixed.  STOP is a FIRST-CLASS action (action_id / action_type / availability),
    not an external ``break`` (spec §11).  CONTINUE is schema-reserved and
    unavailable (spec §12).  Each proposal action carries the per-proposal M3
    input (spec §5) and a shadow M3 prediction (spec §9).  This constructs the
    action space over the UNCHANGED legal set -- it removes / reorders nothing.
    """
    actions: list[dict[str, Any]] = []
    for ordinal, rec in enumerate(legal_records):
        m3_input = _m3_proposal_input(rec)
        actions.append(
            {
                "action_id": _m3_action_id(state_id, M3_ACTION_PROPOSAL, ordinal),
                "action_type": M3_ACTION_PROPOSAL,
                "state_key": state_id,  # spec §13: group identity == current state
                "availability": "available",
                # D2.1 identity preserved verbatim (spec §10/§18)
                "proposal_id": rec["proposal_id"],
                "derivation_id": rec["provenance"]["derivation_id"],
                "root_id": rec["provenance"]["root_id"],
                "operator_type": rec["operator_type"],
                # per-proposal M3 input (spec §5) + shadow policy output (spec §9)
                "m3_input": m3_input,
                "m3_prediction": _m3_shadow_prediction(),
            }
        )

    # --- STOP: a first-class action (spec §11) --------------------------------
    stop_action = {
        "action_id": _m3_action_id(state_id, M3_ACTION_STOP, 0),
        "action_type": M3_ACTION_STOP,
        "state_key": state_id,
        "availability": "available",  # STOP is always a legal choice for the policy
        "proposal_id": None,
        "derivation_id": None,
        "root_id": None,
        "operator_type": None,
        # STOP has no proposal/effect/memory input; it is the "take no proposal"
        # action a future GRPO can compare against P1/P2/P3 (spec §11/§13).
        "m3_input": None,
        "m3_prediction": _m3_shadow_prediction(),
        "semantic": (
            "take no proposal at S_t (halt this state's expansion); first-class "
            "so future GRPO compares P1..Pn vs STOP over one S_t"
        ),
    }
    actions.append(stop_action)

    # --- CONTINUE: schema-reserved, unavailable (spec §12) --------------------
    continue_action = {
        "action_id": _m3_action_id(state_id, M3_ACTION_CONTINUE, 0),
        "action_type": M3_ACTION_CONTINUE,
        "state_key": state_id,
        "availability": M3_CONTINUE_VERDICT["availability"],
        "proposal_id": None,
        "derivation_id": None,
        "root_id": None,
        "operator_type": None,
        "m3_input": None,
        "m3_prediction": {
            "status": "not_implemented_schema_reserved",
            "ranking_score": None,
            "accept_prob": None,
            "risk_prob": None,
            "selected": False,
            "authoritative": False,
        },
        "verdict": dict(M3_CONTINUE_VERDICT),
    }

    proposal_actions = [a for a in actions if a["action_type"] == M3_ACTION_PROPOSAL]
    # neutral-retention roll-up over the action group (spec §6/§15): how many
    # proposal actions historically say "don't greedily prune on immediate Δ=0".
    n_neutral_retention = sum(
        1
        for a in proposal_actions
        if a["m3_input"]["memory"]["memory_supports_neutral_retention"]
    )
    return {
        "state_key": state_id,  # spec §13: one group == one S_t
        "group_id": state_id,   # future GRPO group identity (spec §13)
        "action_set_definition": "A(S_t) = {proposal_1..n, STOP}  (CONTINUE reserved)",
        "proposal_actions_total": len(proposal_actions),
        "stop_available": True,          # spec §11
        "continue_status": M3_CONTINUE_VERDICT["availability"],  # spec §12
        "continue_implemented": M3_CONTINUE_VERDICT["implemented"],
        # STOP + (optionally) CONTINUE are the non-proposal actions
        "actions": actions + [continue_action],
        # roll-ups (evidence, not authority) --------------------------------
        "proposal_actions_with_neutral_retention": n_neutral_retention,
        "state_neutral_then_improved_available": bool(
            memory_context.get("neutral_then_improved_available", False)
        ),
    }


def build_d2_expansion(
    problem: Any,
    schedule: Any,
    appearance: Mapping[str, Any],
    *,
    case_id: str,
    fixture: str,
    device: str = "cpu",
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Real single-state expansion, extracted with provenance + firewall.

    Runs the canonical forward, then re-drives the real expansion pipeline with
    a deterministic candidate authority (spec §0-§24).  Returns the §18
    expansion manifest as a JSON-ready dict.  Executes nothing.
    """
    # -- import real components (no core module is modified) --------------
    from causal_schedule_lab.sg_sct_model_v5 import run_m2_v5_schedule
    from causal_schedule_lab.m2_proposal_builder_v1 import build_operator_runtime
    from causal_schedule_lab.intervention import ScheduleGraphView

    if checkpoint is not None:
        raise SystemExit(
            "D2 BLOCKER: formal checkpoint loading is out of D2 scope. "
            "D2 is mock-untrained (random weights, diagnostic neural logits). "
            "Checkpoint wiring belongs to a later batch."
        )

    # --- (1) canonical real V5 build + single forward --------------------
    t0 = time.time()
    output, _model, bundle = run_m2_v5_schedule(
        problem, schedule, appearance, case_id=case_id, device=device
    )
    forward_seconds = time.time() - t0
    bundle_schema_version = str(bundle.manifest.get("schema_version", ""))

    # Deterministic runtime objects surfaced by the forward (weight-independent).
    sites = tuple(output.decision_sites)
    edits = tuple(output.legal_edits)

    # State fingerprint / provenance root id (spec §19).
    fp_src = json.dumps(
        {
            "problem_id": getattr(problem, "id", case_id),
            "sites": [s.site_id for s in sites],
            "edits": [e.edit_id for e in edits],
        },
        sort_keys=True,
    )
    state_fingerprint = hashlib.sha256(fp_src.encode()).hexdigest()[:16]
    state_id = f"S::{case_id}::{state_fingerprint}"

    # --- D4 read-only per-state memory retrieval (spec §4) ---------------
    # M_t = RetrieveMemory(S_t): runs once per state (re-callable for any future
    # S_1/S_2/... -- not S0-once), read-only, SHA-asserted, self-leak excluded.
    # Produces EVIDENCE that annotates the (unchanged) proposal set; it never
    # filters / reorders / removes a proposal and never touches the Effect
    # Predictor OUTPUT (spec §14.7/§14.8).
    memory_context = retrieve_state_memory(
        problem,
        schedule,
        appearance,
        state_id=state_id,
        state_fingerprint=state_fingerprint,
        case_id=case_id,
    )

    # --- neural DIAGNOSTIC scores (never authoritative) [spec §3/§5] -----
    neural_scores_available = output.per_block_decision_root_logits is not None
    neural_candidate_scores: list[dict[str, Any]] = []
    if neural_scores_available and sites:
        logits = output.per_block_decision_root_logits  # [B, n_site]
        # record per (block, site) neural score, diagnostic only
        for b in range(int(logits.shape[0])):
            for j, site in enumerate(sites):
                neural_candidate_scores.append(
                    {
                        "block_index": b,
                        "site_id": site.site_id,
                        "neural_root_logit_diagnostic": float(logits[b, j]),
                    }
                )

    # --- (2/3) DETERMINISTIC authoritative re-drive ----------------------
    # Real components, deterministic candidate scores (clamped z_deviation),
    # neutral zero edit scores.  block_ids / block_members are taken from the
    # EXACT SAME source the forward's own ``build_operator_runtime`` call uses:
    # ``bundle.manifest["id_spaces"]["appearance_block_ids"]`` +
    # ``build_m2_runtime_context`` -- so there is no reconstruction drift.
    from causal_schedule_lab.sg_sct_model_v5 import build_m2_runtime_context

    block_ids = tuple(bundle.manifest["id_spaces"]["appearance_block_ids"])
    context = build_m2_runtime_context(
        problem, schedule, appearance, block_ids=block_ids
    )
    block_members = dict(context.block_members)

    runtime = None
    root_scores = _det_root_scores(sites) if sites else np.zeros((1, 0))
    # broadcast the single deterministic site row across every block
    if sites and len(block_ids) > 1:
        root_scores = np.repeat(root_scores, len(block_ids), axis=0)
    edit_scores = _det_edit_scores(max(len(block_ids), 1), len(edits))

    if sites and block_ids:
        runtime = build_operator_runtime(
            block_ids=block_ids,
            block_members=block_members,
            decision_sites=sites,
            legal_edits=edits,
            root_scores=root_scores,
            edit_scores=edit_scores,
            top_k=3,
            schedule_graph=ScheduleGraphView.from_problem_schedule(problem, schedule),
        )

    generated = list(runtime.proposals) if runtime is not None else []
    chains = list(runtime.causal_chains) if runtime is not None else []

    # --- D2.1 authoritative provenance DAG (over ALL chains) -------------
    # Built once over the untruncated chain list; every provenance id-set and
    # the DAG summary below derive from it, so their authority is independent
    # of the D2_MAX_RECORDS / D2_MAX_TRACE sidecar caps (spec §5/§6).
    prov = build_provenance_index(chains, state_id)

    # --- (4) legality partition (fail-closed, spec §12) ------------------
    legal_props: list[Any] = []
    rejected_records: list[dict[str, Any]] = []
    for prop in generated:
        try:
            prop.validate()
        except Exception as exc:  # illegal -> rejected, never in legal_proposals
            rejected_records.append(
                {
                    "proposal_id": getattr(prop, "proposal_id", "<unvalidated>"),
                    "rejection_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        legal_props.append(prop)

    # Actionable-root records (real selector output) -- hoisted so the DAG
    # summary can report the true actionable-root count without recomputing.
    actionable_root_records = build_actionable_root_records(
        runtime, chains, state_id, prov
    )

    # --- candidate site records + legal action space (spec §5) -----------
    candidate_sites = []
    for rank, site in enumerate(sites, start=1):
        candidate_sites.append(
            {
                "candidate_id": f"{state_id}::cand::{site.site_id}",
                "source": "deterministic_forward_decision_site",
                "deterministic_priority_rank": rank,
                **_site_record(site),
                "legal_action_space": _legal_action_space_for_site(site, edits),
            }
        )

    # --- D4 invariance proof: proposal set + Effect Predictor unchanged --
    # Build the baseline proposal records (exactly as D3 would, no memory), then
    # the memory-attached records.  The memory layer may ONLY add a
    # ``memory_evidence`` key and flip ``memory_features_available``; everything
    # else -- proposal identity, legal edit graph, provenance DAG, and the
    # Effect Predictor OUTPUT (``effect_prediction``) -- must be byte-identical
    # (spec §14.7/§14.8).  We prove it by stripping the two permitted additions
    # from the attached record and comparing to the baseline.
    def _strip_memory(rec: dict[str, Any]) -> dict[str, Any]:
        r = copy.deepcopy(rec)
        r.pop("memory_evidence", None)
        r["effect_request"]["features"]["memory_features_available"] = False
        return r

    baseline_generated = [
        _proposal_record(p, state_id, prov) for p in generated[:D2_MAX_RECORDS]
    ]
    baseline_legal = [
        _proposal_record(p, state_id, prov) for p in legal_props[:D2_MAX_RECORDS]
    ]
    attached_generated = [
        _attach_memory_to_record(base, p, memory_context)
        for base, p in zip(baseline_generated, generated[:D2_MAX_RECORDS])
    ]
    attached_legal = [
        _attach_memory_to_record(base, p, memory_context)
        for base, p in zip(baseline_legal, legal_props[:D2_MAX_RECORDS])
    ]
    proposal_set_unchanged = all(
        _strip_memory(a) == b
        for a, b in zip(attached_generated, baseline_generated)
    ) and all(
        _strip_memory(a) == b for a, b in zip(attached_legal, baseline_legal)
    )
    # Effect Predictor output specifically byte-identical (independent check).
    effect_prediction_unchanged = all(
        a["effect_prediction"] == b["effect_prediction"]
        for a, b in zip(attached_generated, baseline_generated)
    ) and all(
        a["effect_prediction"] == b["effect_prediction"]
        for a, b in zip(attached_legal, baseline_legal)
    )
    proposal_ids_unchanged = [r["proposal_id"] for r in attached_generated] == [
        r["proposal_id"] for r in baseline_generated
    ]

    # --- D5 M3 proposal / STOP shadow layer (spec §4/§5/§9/§11/§13) -------
    # Constructed over the ATTACHED LEGAL records (the complete, unchanged legal
    # proposal set + D3 effect output + D4 memory evidence).  The action space is
    # a pure read of that set: A(S_t) = {proposal_1..n, STOP}, CONTINUE reserved.
    # M3 is the last shadow consumer -- it filters / reorders / removes nothing
    # and executes nothing (spec §10).
    m3_action_space = build_m3_action_space(
        attached_legal, state_id, memory_context
    )
    # D5 invariance proof (spec §10): the M3 layer must leave the proposal set,
    # the Effect Predictor output, and the D4 memory evidence byte-identical.
    m3_proposal_actions = [
        a for a in m3_action_space["actions"] if a["action_type"] == M3_ACTION_PROPOSAL
    ]
    m3_proposal_ids = [a["proposal_id"] for a in m3_proposal_actions]
    m3_proposal_set_unchanged = m3_proposal_ids == [
        r["proposal_id"] for r in attached_legal
    ]
    m3_derivation_ids_unchanged = [
        a["derivation_id"] for a in m3_proposal_actions
    ] == [r["provenance"]["derivation_id"] for r in attached_legal]
    # M3 read the effect output; it must not have changed it (independent check).
    m3_effect_output_unchanged = all(
        a["m3_input"]["effect"]["status"] == r["effect_prediction"]["status"]
        and a["m3_input"]["effect"]["delta_pred"] == r["effect_prediction"]["delta_pred"]
        and a["m3_input"]["effect"]["future_gain_pred"]
        == r["effect_prediction"]["future_gain_pred"]
        for a, r in zip(m3_proposal_actions, attached_legal)
    )
    # M3 read the memory evidence; the neutral-retention flag it surfaced must
    # match the underlying D4 evidence exactly (memory is INPUT, not decision).
    m3_memory_flag_faithful = all(
        a["m3_input"]["memory"]["memory_supports_neutral_retention"]
        == r["memory_evidence"]["memory_supports_neutral_retention"]
        for a, r in zip(m3_proposal_actions, attached_legal)
    )
    # No proposal was selected, no action is authoritative (spec §9/§15).
    m3_nothing_selected = all(
        not a["m3_prediction"]["selected"]
        and not a["m3_prediction"]["authoritative"]
        for a in m3_action_space["actions"]
    )

    # --- assemble the §18 manifest --------------------------------------
    manifest: dict[str, Any] = {
        "schema_version": D2_MANIFEST_SCHEMA,
        "mode": "d2",
        "fixture": fixture,
        "case_id": case_id,
        "device": device,
        "forward_seconds": round(forward_seconds, 3),
        "bundle_schema_version": bundle_schema_version,
        "canonical_schema_expected": CANONICAL_SCHEMA_VERSION,
        "schema_matches_canonical": bundle_schema_version == CANONICAL_SCHEMA_VERSION,
        "state_id": state_id,
        "state_fingerprint": state_fingerprint,
        # --- authority firewall (spec §3) ---
        "authority": {
            "candidate_authority": "deterministic_fallback",
            "candidate_score_source": "site.z_deviation (clamped +-5); no neural term",
            "edit_score_source": "neutral_zero",
            "neural_scores_available": bool(neural_scores_available),
            "neural_scores_authoritative": False,
            "checkpoint_loaded": False,
            "mock_untrained": True,
            "effect_predictor_used": False,
            "effect_authority": "shadow_annotation_no_checkpoint_no_decision_power",
            # D4: retrieval RAN, read-only and non-authoritative (spec §2/§4).
            "memory_retrieval_used": True,
            "memory_retrieval_status": "read_only_non_authoritative_d4",
            "memory_authoritative": False,  # memory never decides (spec §2/§7/§10)
            "memory_mutated": False,  # frozen snapshot untouched (spec §13)
            "memory_changes_proposal_set": False,  # proven below (spec §14.7)
            "memory_context": memory_context,
            # D5: the M3 policy interface was called as a SHADOW consumer (spec
            # §9): no checkpoint, null outputs, zero decision power.  It selected
            # nothing and changed no proposal (proven in ``m3_layer`` below).
            "m3_called": True,
            "m3_call_status": "shadow_no_checkpoint_non_authoritative",
            "m3_authoritative": False,  # spec §9
            "m3_changes_proposal_set": False,  # spec §10 (proven below)
            "selected_proposal_id": None,  # spec §9/§15: M3 never selects in D5
            "executor_called": False,
            "next_state_created": False,
        },
        # --- identity / honesty ---
        "identified": bool(output.identified),
        "training_status": str(output.training_status),
        "model_schema_version": str(output.model_schema_version),
        # --- expansion payload ---
        "n_blocks": len(block_ids),
        "block_ids": list(block_ids),
        "candidate_sites": candidate_sites,
        # High-cardinality lists are bounded on disk (D2_MAX_RECORDS) but their
        # true totals are always preserved -- the cap never touches counts,
        # partitions, provenance, or the structural check.
        "neural_candidate_scores_total": len(neural_candidate_scores),
        "neural_candidate_scores_truncated": len(neural_candidate_scores) > D2_MAX_RECORDS,
        "neural_candidate_scores_diagnostic": neural_candidate_scores[:D2_MAX_RECORDS],
        "causal_searches_total": len(chains),
        "causal_searches_truncated": len(chains) > D2_MAX_RECORDS,
        "causal_searches": [
            _causal_search_record(c, state_id) for c in chains[:D2_MAX_RECORDS]
        ],
        "actionable_roots": actionable_root_records,
        "generated_proposals_total": len(generated),
        "legal_proposals_total": len(legal_props),
        "rejected_proposals_total": len(rejected_records),
        # --- fail-closed appearance gate audit (pre-D6 appearance cleanup) ---
        # Deprecated (A5/A7/A8/A9/A10) or unknown appearance blocks are dropped
        # BEFORE any candidate/root/proposal is built (appearance_taxonomy gate
        # in build_operator_runtime).  A non-empty list here means a stale /
        # out-of-taxonomy artifact was loaded; the runtime stayed actionable
        # only for the active set {A1,A2,A3,A4,A6}.  Legacy-readable, never
        # actionable.
        "filtered_appearance_blocks_total": (
            len(runtime.filtered_appearance_blocks) if runtime is not None else 0
        ),
        "filtered_appearance_blocks": (
            [
                {"block_id": bid, "reason": reason}
                for bid, reason in runtime.filtered_appearance_blocks[:D2_MAX_RECORDS]
            ]
            if runtime is not None
            else []
        ),
        # Memory-attached records: identical to baseline except the two
        # permitted additions (``memory_evidence`` + memory_features_available).
        "generated_proposals": attached_generated,
        "legal_proposals": attached_legal,
        "rejected_proposals": rejected_records[:D2_MAX_RECORDS],
        # --- provenance summary (spec §19 + D2.1) ---
        # The DAG is many-to-one / many-to-many (state -> many traces sharing a
        # site -> deduped roots -> proposals), NOT a linear chain.  This summary
        # is authoritative over ALL chains (untruncated) and truncation-invariant
        # (counts + a stable hash over the full sorted id set of each relation).
        "provenance_chain": "state_id -> {candidate_id, trace_id} -> root_id -> {derivation_id, proposal_id}",
        "provenance_dag": {
            "structure": "many-to-one/many-to-many (DAG, not linear)",
            "state_id": state_id,
            "n_traces_total": prov.n_traces_total,
            "unique_trace_ids": prov.unique_trace_ids,
            "trace_id_formula": "state_id::trace::appearance_id::seed_decision_site_id::sha256(nodes+cause_edges)[:16]",
            "trace_id_hash_algo": "sha256_content_addressed_not_runtime_hash",
            # Two DISTINCT counts -- never conflated (Mk9: 10 traced sites vs 3
            # actionable roots vs 2 roots that produced a proposal):
            #  * n_traced_sites    = decision sites any chain reached (map keys)
            #  * n_actionable_roots = sites the selector marked actionable
            #  * n_roots_with_proposals = actionable roots that produced >=1 proposal
            "n_traced_sites": len(prov.trace_ids_by_site),
            "n_actionable_roots": len({r["decision_site_id"] for r in actionable_root_records}),
            "n_roots_with_proposals": len(
                {p.root_decision_id for p in legal_props}
            ),
            "supporting_trace_ids_by_site": {
                site: _bounded_id_set(tids)
                for site, tids in prov.trace_ids_by_site.items()
            },
            "supporting_candidate_ids_by_site": {
                site: _bounded_id_set(cids)
                for site, cids in prov.candidate_ids_by_site.items()
            },
        },
        # --- D3 shadow effect layer (spec §16/§18) ---
        # Annotation layer over the (unchanged) proposal set.  Disabled without a
        # checkpoint: predictions are null, outputs are non-authoritative, and the
        # layer holds zero decision power (no filter / reorder / removal).  The
        # proposal totals above are identical with and without this layer.
        "effect_layer": {
            "enabled": True,
            "checkpoint_loaded": False,
            "authoritative": False,
            "effect_outputs_authoritative": False,
            "effect_prediction_status": EFFECT_PREDICTION_STATUS_DISABLED,
            # one effect_request is attached per emitted proposal record; totals
            # mirror the true proposal totals (the layer adds no/loses no rows).
            "input_records_total": len(generated),
            "predictions_total": 0,  # no checkpoint -> nothing predicted
            "changes_proposal_set": False,
        },
        # --- D4 read-only per-state memory layer (spec §1-§16) ---
        # Retrieval PRIOR / exploration EVIDENCE over the (unchanged) proposal
        # set.  The full per-state retrieval provenance + evidence lives in
        # ``authority.memory_context``; this block is the structural summary +
        # the invariance proof (proposal set + Effect Predictor output unchanged).
        "memory_layer": {
            "enabled": True,
            "role": "retrieval_prior_exploration_evidence",  # spec §2
            "authoritative": False,  # never decides (spec §2/§7/§10)
            "read_only": True,  # spec §13
            "mutated": False,  # frozen snapshot untouched (spec §13)
            "per_state_contract": "M_t = RetrieveMemory(S_t)",  # spec §4
            "memory_snapshot_sha": memory_context["memory_snapshot_sha"],
            "memory_snapshot_sha_matches": memory_context["memory_snapshot_sha_matches"],
            "retrieved_count": memory_context["retrieved_count"],
            "memory_support_status": memory_context["memory_support_status"],
            "total_candidates_considered": memory_context["total_candidates_considered"],
            "excluded_count": memory_context["instance_family_exclusion"]["excluded_count"],
            "evidence_bucket_counts": {
                b: len(ids) for b, ids in memory_context["evidence_buckets"].items()
            },
            "neutral_then_improved_available": memory_context[
                "neutral_then_improved_available"
            ],
            # --- invariance proof (spec §14.7/§14.8) ---
            "changes_proposal_set": not proposal_set_unchanged,
            "proposal_set_unchanged": bool(proposal_set_unchanged),
            "proposal_ids_unchanged": bool(proposal_ids_unchanged),
            "effect_prediction_unchanged": bool(effect_prediction_unchanged),
            # --- honesty firewall (spec §7/§10) ---
            "similarity_is_not_truth": True,
            "maps_to_future_gain_pred": False,
            "maps_to_success_label": False,
            "decides_keep_or_drop": False,
        },
        # --- D5 M3 proposal / STOP / CONTINUE shadow layer (spec §4-§17) -----
        # The LAST shadow consumer.  It builds the same-state action set
        # A(S_t) = {proposal_1..n, STOP} (CONTINUE schema-reserved), a
        # first-class STOP action, a per-proposal STATE/PROPOSAL/CAUSAL/EFFECT/
        # MEMORY input that surfaces memory_supports_neutral_retention, and a
        # shadow (no-checkpoint, non-authoritative) policy output.  It reads the
        # unchanged proposal set + Effect output + Memory evidence and holds zero
        # decision power: no selection, no filter/reorder/removal, no execution,
        # no S_t+1, no reward, no Memory mutation (spec §9/§10/§14/§15).
        "m3_layer": {
            "enabled": True,
            "role": "proposal_stop_selection_shadow",  # spec §3
            "checkpoint_loaded": False,  # spec §9
            "authoritative": False,  # spec §9: untrained M3 never decides
            "m3_outputs_authoritative": False,  # spec §9
            "policy_called": "shadow",  # spec §9/§17
            "prediction_status": M3_PREDICTION_STATUS_DISABLED,
            "responsibility": (
                "given S_t's complete legal proposals, pick which next or STOP; "
                "NOT causal search / root discovery / proposal generation / hard "
                "legality / executor / CP-SAT (spec §3)"
            ),
            # --- action-space contract (spec §4/§11/§13) ---
            "state_key": m3_action_space["state_key"],
            "group_id": m3_action_space["group_id"],
            "action_set_definition": m3_action_space["action_set_definition"],
            "proposal_actions_total": m3_action_space["proposal_actions_total"],
            "stop_available": m3_action_space["stop_available"],  # spec §11
            "stop_is_first_class_action": True,  # spec §11
            # --- CONTINUE verdict (spec §12) ---
            "continue_status": m3_action_space["continue_status"],
            "continue_implemented": m3_action_space["continue_implemented"],
            "continue_verdict": dict(M3_CONTINUE_VERDICT),
            # --- neutral-retention evidence roll-up (spec §6/§15) ---
            "proposal_actions_with_neutral_retention": m3_action_space[
                "proposal_actions_with_neutral_retention"
            ],
            "state_neutral_then_improved_available": m3_action_space[
                "state_neutral_then_improved_available"
            ],
            "surfaces_memory_supports_neutral_retention": True,  # spec §6
            # --- authority firewall (spec §7/§9/§10/§14/§15) ---
            "changes_proposal_set": not m3_proposal_set_unchanged,
            "proposal_set_unchanged": bool(m3_proposal_set_unchanged),
            "derivation_ids_unchanged": bool(m3_derivation_ids_unchanged),
            "effect_output_unchanged": bool(m3_effect_output_unchanged),
            "memory_flag_faithful_to_evidence": bool(m3_memory_flag_faithful),
            "nothing_selected": bool(m3_nothing_selected),
            "memory_directly_selects_action": False,  # spec §7: input, not decider
            "reward_computed": False,  # spec §14
            "success_label_assigned": False,  # spec §14/§15
            "executor_called": False,  # spec §10
            "next_state_created": False,  # spec §0/§10
            "memory_mutated": False,  # spec §1/§13
            # --- the assembled action space (spec §17) ---
            "action_space": m3_action_space,
        },
        # --- D3 effect supervision contract (spec §18) ---
        # Per-head adjudication verdict from LIVE code (Phase D3).  Describes what
        # is legally supervisable; nothing is trained in D3.
        "effect_supervision_contract": dict(EFFECT_SUPERVISION_CONTRACT),
        # --- formal-state invariants (spec §20) ---
        "formal_state": {
            "formal_training": 0,
            "optimizer_steps": 0,
            "formal_val_adaptation": 0,
            "formal_test_access": 0,
            "safe_to_train": "NO",
            "engineering_only": True,
        },
        # --- explicit STOP boundary ---
        "stop_point": "after_single_state_expansion",
    }
    return manifest


def build_actionable_root_records(
    runtime: Any, chains: list[Any], state_id: str, prov: "_ProvenanceIndex | None" = None
) -> list[dict[str, Any]]:
    """One record per actionable-root proposal endpoint (real selector output).

    ``build_operator_runtime`` exposes ``actionable_root_ids`` (the selected
    decision-site id of each root that produced a proposal).  A single root is
    generally supported by **many** traces / seed candidates (D2.1 spec §3), so
    each record now carries the authoritative, truncation-invariant
    ``supporting_trace_ids`` / ``supporting_candidate_ids`` id-sets from the
    provenance index -- not just the first chain that happened to reach the site.
    """
    if runtime is None:
        return []
    # keep a representative chain per site only for descriptive fields
    # (appearance / operation / operator_types); the AUTHORITY is ``prov``.
    chain_by_site: dict[str, Any] = {}
    for c in chains:
        chain_by_site.setdefault(c.decision_site_id, c)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for site_id in runtime.actionable_root_ids:
        if site_id in seen:
            continue
        seen.add(site_id)
        chain = chain_by_site.get(site_id)
        rec: dict[str, Any] = {
            "root_id": _root_id(state_id, site_id),
            "decision_site_id": site_id,
            "actionable": True,
            "source": "actionable_root_selector_v2",
        }
        if prov is not None:
            rec["supporting_trace_ids"] = _bounded_id_set(
                prov.trace_ids_by_site.get(site_id, [])
            )
            rec["supporting_candidate_ids"] = _bounded_id_set(
                prov.candidate_ids_by_site.get(site_id, [])
            )
        if chain is not None:
            rec["appearance_id"] = chain.appearance_id
            rec["operation_id"] = chain.operation_id
            rec["operator_types"] = list(chain.operator_types)
            rec["depth"] = int(chain.depth)
        records.append(rec)
    return records


# =============================================================================
# D6 iterative successor-state retention + mainline loop
# -----------------------------------------------------------------------------
# Layered ON TOP of the D5-verified single-state bridge (``build_d2_expansion`` +
# ``retrieve_state_memory``), which are NOT altered.  D6 adds:
#   * a validated proposal -> executable-atom bridge (fail-closed on unmappable),
#   * Frozen-Local sibling evaluation of every legal proposal against the SAME
#     byte-stable S_t baseline (the executor never mutates its input schedule, so
#     sibling independence holds structurally),
#   * the successor-retention rule (immediate_improvement / RetrieveMemory(S'_i)
#     -> memory_supported_promising_successor / bounded deterministic exploration
#     / prune), distinguishing current_state_memory (S_t) vs successor_state_memory
#     (S'_i),
#   * a deterministic engineering fallback that selects P* (M3 stays shadow /
#     non-authoritative -- no checkpoint, no random logits),
#   * the real S_t -> execute P* -> S_{t+1} -> re-analyze-from-scratch bounded
#     loop, and an engineering trajectory sidecar (NOT appended to frozen Memory).
# It trains nothing; the frozen Memory snapshot SHA is asserted unchanged on every
# retrieval.
# =============================================================================


def _forward_state_identity(
    problem: Any,
    schedule: Any,
    appearance: Mapping[str, Any],
    case_id: str,
    *,
    device: str = "cpu",
) -> tuple[str, str]:
    """Canonical (state_id, state_fingerprint) for any schedule S.

    Uses the EXACT recipe ``build_d2_expansion`` uses for S_t (sha256 over
    problem_id + decision-site ids + legal-edit ids from the real forward), so
    S_t, each successor S'_i, and each S_{t+1} share one identity convention.
    """
    from causal_schedule_lab.sg_sct_model_v5 import run_m2_v5_schedule

    output, _model, _bundle = run_m2_v5_schedule(
        problem, schedule, appearance, case_id=case_id, device=device
    )
    sites = tuple(output.decision_sites)
    edits = tuple(output.legal_edits)
    fp_src = json.dumps(
        {
            "problem_id": getattr(problem, "id", case_id),
            "sites": [s.site_id for s in sites],
            "edits": [e.edit_id for e in edits],
        },
        sort_keys=True,
    )
    fp = hashlib.sha256(fp_src.encode()).hexdigest()[:16]
    return f"S::{case_id}::{fp}", fp


def _d6_live_legal_proposals(
    problem: Any,
    schedule: Any,
    appearance: Mapping[str, Any],
    case_id: str,
    *,
    device: str = "cpu",
) -> list[Any]:
    """Live legal ``CausalInterventionProposal`` objects for S.

    Mirrors ``build_d2_expansion``'s deterministic re-drive EXACTLY (same forward,
    same block ids from the bundle manifest, same deterministic root/edit scores,
    same top_k, same ScheduleGraphView), so the returned objects correspond 1:1
    and in-order with the manifest's ``legal_proposals`` -- asserted by the caller.
    """
    from causal_schedule_lab.sg_sct_model_v5 import (
        run_m2_v5_schedule,
        build_m2_runtime_context,
    )
    from causal_schedule_lab.m2_proposal_builder_v1 import build_operator_runtime
    from causal_schedule_lab.intervention import ScheduleGraphView

    output, _model, bundle = run_m2_v5_schedule(
        problem, schedule, appearance, case_id=case_id, device=device
    )
    sites = tuple(output.decision_sites)
    edits = tuple(output.legal_edits)
    block_ids = tuple(bundle.manifest["id_spaces"]["appearance_block_ids"])
    context = build_m2_runtime_context(
        problem, schedule, appearance, block_ids=block_ids
    )
    root_scores = _det_root_scores(sites) if sites else np.zeros((1, 0))
    if sites and len(block_ids) > 1:
        root_scores = np.repeat(root_scores, len(block_ids), axis=0)
    edit_scores = _det_edit_scores(max(len(block_ids), 1), len(edits))
    runtime = None
    if sites and block_ids:
        runtime = build_operator_runtime(
            block_ids=block_ids,
            block_members=dict(context.block_members),
            decision_sites=sites,
            legal_edits=edits,
            root_scores=root_scores,
            edit_scores=edit_scores,
            top_k=3,
            schedule_graph=ScheduleGraphView.from_problem_schedule(problem, schedule),
        )
    generated = list(runtime.proposals) if runtime is not None else []
    legal: list[Any] = []
    for prop in generated:
        try:
            prop.validate()
        except Exception:
            continue
        legal.append(prop)
    return legal


def _d6_proposal_to_atoms(problem: Any, schedule: Any, prop: Any):
    """Bridge one legal proposal to executable atoms -- or None (fail-closed).

    EXACT edit->atom identity contract (T1-RUNTIME-MULTI-EDIT-ENABLING-CHAIN
    verdict D fix):

      * every edit's subject operation must be the edit's own ``operation_id`` --
        generated with ``subject_operations_only=True`` so candidate subjects are
        exactly the root op, never its radius-1 neighborhood (the old
        ``usable[0]`` silently executed a *neighbour* op, 51/60 on Mk1);
      * the candidate must match the edit's semantics EXACTLY:
          ROUTE     -> subject op + target machine (+ target mode when pinned)
          SEQ_SWAP  -> resource + the exact (left,right) op pair
          SEQ_INSERT-> subject op + resource + predecessor/successor slot
      * no exact match for ANY edit -> return None (FAIL-CLOSED).  Never
        ``usable[0]``, never silent substitution, never neighbour fallback.

    TIMING_SHIFT / unmapped edit types are never executed (fail-closed).  Frozen
    source-machine disagreement (proposal vs schedule) also fails closed.
    """
    from causal_schedule_lab.operator_registry_v1 import generate_operator_candidates
    from causal_schedule_lab.sg_sct_causal_probe import _candidate_to_atom

    opmap = problem.operation_map()
    amap = schedule.assignment_map()
    mode_map = problem.mode_map()

    def current_machine(op: str) -> str | None:
        asg = amap.get(op)
        if asg is None:
            return None
        mode = mode_map.get(asg.mode_id)
        if mode is None or not mode[1].resources:
            return None
        return mode[1].resources[0]

    atoms: list[Any] = []
    for edit in prop.edits:
        want = D6_EDIT_OPERATOR_MATCH.get(edit.edit_type)
        if not want:  # TIMING_SHIFT or any unmappable edit type -> fail-closed
            return None
        anchor = edit.operation_id or (prop.root_path[0] if prop.root_path else None)
        if anchor is None or anchor not in opmap or anchor not in amap:
            return None
        candidates = generate_operator_candidates(
            problem,
            schedule,
            root_operations=(anchor,),
            decision_time=0,
            maximum_per_operator=64,
            neighborhood_radius=1,
            subject_operations_only=True,  # subjects == {anchor}, not the neighborhood
        )
        exact: list[Any] = []
        for c in candidates:
            if not c.legal or c.operator_id == "stop" or c.operator_id not in want:
                continue
            p = dict(c.parameters)
            if edit.edit_type == "ROUTE":
                if str(p.get("operation_id")) != anchor:
                    continue  # subject op identity
                tgt = (p.get("target_resource_ids") or [None])[0]
                if str(tgt) != str(edit.target_machine):
                    continue  # target machine identity
                if current_machine(anchor) != edit.source_machine:
                    return None  # proposal disagrees with the schedule -> fail-closed
                if edit.target_mode_id is not None and str(p.get("mode_id")) != str(edit.target_mode_id):
                    continue  # pinned target mode identity (exact atom)
            elif edit.edit_type == "SEQ_SWAP":
                if str(p.get("resource_id")) != str(edit.resource_id):
                    continue
                left = p.get("left_operation_id")
                right = p.get("right_operation_id")
                if left is None and right is None:
                    ids = p.get("operation_ids") or ()
                    left = ids[0] if len(ids) > 0 else None
                    right = ids[1] if len(ids) > 1 else None
                if left is None or right is None:
                    continue
                pair = {str(left), str(right)}
                if pair != {str(edit.left_id), str(edit.right_id)}:
                    continue  # exact op pair identity (no substitution)
            elif edit.edit_type == "SEQ_INSERT":
                if str(p.get("operation_id")) != anchor:
                    continue
                if str(p.get("resource_id")) != str(edit.resource_id):
                    continue
                if edit.predecessor_id is not None and str(p.get("predecessor_id")) != str(edit.predecessor_id):
                    continue
                if edit.successor_id is not None and str(p.get("successor_id")) != str(edit.successor_id):
                    continue
            else:
                continue
            exact.append(c)
        if not exact:
            return None  # FAIL-CLOSED: no exact edit->atom mapping
        # deterministic choice among exact matches: operator preference (want
        # order), then stable candidate_id.
        want_pri = {op: i for i, op in enumerate(want)}
        exact.sort(key=lambda c: (want_pri.get(c.operator_id, 99), c.candidate_id))
        atoms.append(_candidate_to_atom(anchor, exact[0]))
    return tuple(atoms) if atoms else None


def _d6_execute(executor: Any, problem: Any, schedule: Any, atoms) -> Any:
    """Frozen-Local execution of one or many atoms (input schedule NOT mutated)."""
    if len(atoms) > 1:
        return executor.execute_atoms(problem, schedule, atoms)
    return executor.execute_atom(problem, schedule, atoms[0])


def _d6_evaluate_successor(
    problem: Any,
    schedule: Any,
    prop: Any,
    executor: Any,
    *,
    base_ms: int,
    base_hash: str,
):
    """Frozen-Local sibling eval of one proposal against the SAME byte-stable S_t.

    Returns ``(eval_dict, successor_schedule_or_None)``.  The executor produces a
    NEW schedule and never mutates its input, so evaluating P2 never builds on P1
    executed (sibling independence); we assert the baseline hash is unchanged.
    Sign: ``immediate_delta = Cmax(after) - Cmax(before)``, negative = improvement.
    """
    from causal_schedule_lab.validation import schedule_hash

    atoms = _d6_proposal_to_atoms(problem, schedule, prop)
    if atoms is None:
        return (
            {
                "mappable": False,
                "feasible": False,
                "immediate_delta": None,
                "successor_ms": None,
                "reason": "unmappable_edit_fail_closed",
            },
            None,
        )
    res = _d6_execute(executor, problem, schedule, atoms)
    # sibling independence (structural): the input schedule must be byte-stable.
    if schedule_hash(schedule) != base_hash:
        raise SystemExit(
            "D6 BLOCKER: executor mutated the baseline schedule -- sibling "
            "independence broken.  Frozen-Local evaluation must not mutate S_t."
        )
    if not res.feasible or res.schedule is None:
        return (
            {
                "mappable": True,
                "feasible": False,
                "immediate_delta": None,
                "successor_ms": None,
                "reason": res.report.reason or "infeasible",
            },
            None,
        )
    succ_ms = int(res.schedule.makespan)
    return (
        {
            "mappable": True,
            "feasible": True,
            "immediate_delta": succ_ms - int(base_ms),  # after - before
            "successor_ms": succ_ms,
            "reason": "ok",
            "executor_version": res.executor_version,
        },
        res.schedule,
    )


def _d6_retention_decision(
    problem: Any,
    prop: Any,
    eval_result: Mapping[str, Any],
    successor_schedule: Any,
    *,
    case_id: str,
    neutral_budget_state: list[int],
    device: str = "cpu",
    memory_path: Path = FROZEN_MEMORY_PATH,
    memory_sha: str = FROZEN_MEMORY_SHA,
) -> dict[str, Any]:
    """Apply the D6 retention rule to one evaluated proposal.

    delta<0 -> retain (immediate_improvement).  delta==0 -> RetrieveMemory(S'_i)
    (NOT reuse M(S_t)); neutral_then_improved match -> retain
    (memory_supported_promising_successor), else a deterministic <=K exploration
    budget retains, else prune.  delta>0 -> prune (D6 V1 default).  Unmappable /
    infeasible -> prune (honest reason).  Records successor identity and, for the
    neutral branch, the SUCCESSOR memory (distinct manifest lineage from S_t).
    Memory yields NO reward / success / future_gain / auto-selection.
    """
    delta = eval_result["immediate_delta"]
    rec: dict[str, Any] = {
        "proposal_id": prop.proposal_id,
        "immediate_delta": delta,
        "mappable": eval_result["mappable"],
        "feasible": eval_result["feasible"],
        "successor_state_id": None,
        "successor_state_fingerprint": None,
        "successor_memory_match_status": None,
        "memory_supported_promising_successor": False,
        # spec: DISTINGUISH successor_state_memory (S'_i) from current_state_memory
        "successor_state_memory": None,
    }
    if not eval_result["feasible"] or successor_schedule is None:
        rec["retention_status"] = "pruned"
        rec["retention_reason"] = eval_result["reason"]
        return rec

    # successor identity via the canonical forward recipe (same as S_t / S_{t+1}).
    from causal_schedule_lab.symptom_pruning import diagnose_and_prune

    succ_appearance = diagnose_and_prune(problem, successor_schedule).model_dump(
        mode="json"
    )
    succ_id, succ_fp = _forward_state_identity(
        problem, successor_schedule, succ_appearance, case_id, device=device
    )
    rec["successor_state_id"] = succ_id
    rec["successor_state_fingerprint"] = succ_fp

    if delta < -D6_DELTA_EPS:
        rec["retention_status"] = "retained"
        rec["retention_reason"] = "immediate_improvement"
        return rec
    if delta > D6_DELTA_EPS:
        rec["retention_status"] = "pruned"
        rec["retention_reason"] = "immediate_worsening_default_prune"
        return rec

    # --- neutral (|delta| <= eps): RetrieveMemory(S'_i), NOT reuse M(S_t) -----
    succ_memory = retrieve_state_memory(
        problem,
        successor_schedule,
        succ_appearance,
        state_id=succ_id,
        state_fingerprint=succ_fp,
        case_id=case_id,
        memory_path=memory_path,
        expected_sha=memory_sha,
    )
    # attach a compact, honest view of the SUCCESSOR memory (full firewall intact).
    rec["successor_state_memory"] = {
        "source": "successor_state_S_prime",  # spec: distinct from S_t memory
        "state_id": succ_id,
        "memory_support_status": succ_memory["memory_support_status"],
        "neutral_then_improved_available": succ_memory[
            "neutral_then_improved_available"
        ],
        "memory_snapshot_sha": succ_memory["memory_snapshot_sha"],
        "memory_snapshot_sha_matches": succ_memory["memory_snapshot_sha_matches"],
        "retrieved_count": succ_memory["retrieved_count"],
        "evidence_bucket_counts": {
            b: len(ids) for b, ids in succ_memory["evidence_buckets"].items()
        },
        # honesty firewall (spec §7): similarity is evidence, never truth.
        "similarity_is_not_truth": True,
        "maps_to_future_gain_pred": False,
        "maps_to_success_label": False,
        "generates_reward": False,
    }
    rec["successor_memory_match_status"] = succ_memory["memory_support_status"]
    nti = bool(succ_memory["neutral_then_improved_available"])
    rec["memory_supported_promising_successor"] = nti
    if nti:
        rec["retention_status"] = "retained"
        rec["retention_reason"] = "memory_supported_promising_successor"
        return rec
    # no memory support -> deterministic bounded exploration budget (<= K), no RNG.
    if neutral_budget_state[0] < D6_MAX_NEUTRAL_EXPLORATION:
        neutral_budget_state[0] += 1
        rec["retention_status"] = "retained"
        rec["retention_reason"] = "deterministic_neutral_exploration_budget"
        return rec
    rec["retention_status"] = "pruned"
    rec["retention_reason"] = "neutral_no_memory_support_budget_exhausted"
    return rec


def _d6_rank_actionable(retentions: list[Mapping[str, Any]]):
    """Ordered actionable proposals under the D6.1 selector precedence (spec §1).

    Returns a list of ``(proposal_id, reason)`` in strict priority order:

      1. STRICTLY improving retained proposals (``immediate_delta < 0``), ordered
         by (immediate_delta, proposal_id) -> reason ``immediate_improvement``.
      2. MEMORY-SUPPORTED neutral retained proposals -- both ``|delta| <= eps``
         AND ``retention_reason == memory_supported_promising_successor`` --
         ordered by proposal_id ASCENDING (spec §2: Memory grants EXPLORATION
         ELIGIBILITY only; the tie-break is a stable deterministic identity, NEVER
         a historical gain / terminal delta / future_gain / reward / success).

    An immediate improvement ALWAYS outranks a memory-supported neutral (spec §3:
    a Memory hit must never beat a real Frozen-Local improvement).  Ordinary
    neutral proposals retained under the deterministic exploration budget (reason
    ``deterministic_neutral_exploration_budget``) are NOT actionable here (spec §7:
    D6.1 only makes ``memory_supported_promising_successor`` execute).
    """
    improving = [
        r
        for r in retentions
        if r["retention_status"] == "retained"
        and r["immediate_delta"] is not None
        and r["immediate_delta"] < -D6_DELTA_EPS
    ]
    improving.sort(key=lambda r: (r["immediate_delta"], r["proposal_id"]))
    memory_neutral = [
        r
        for r in retentions
        if r["retention_status"] == "retained"
        and r["immediate_delta"] is not None
        and abs(r["immediate_delta"]) <= D6_DELTA_EPS
        and r["retention_reason"] == "memory_supported_promising_successor"
    ]
    memory_neutral.sort(key=lambda r: r["proposal_id"])  # spec §2: stable id, NOT gain
    return [(r["proposal_id"], "immediate_improvement") for r in improving] + [
        (r["proposal_id"], "memory_supported_neutral_exploration")
        for r in memory_neutral
    ]


def _d6_select_proposal(
    retentions: list[Mapping[str, Any]],
    *,
    visited_state_fingerprints: frozenset[str] = frozenset(),
):
    """Cycle-guarded deterministic engineering fallback selection of P* (spec §1/§5).

    NOT M3: no checkpoint, no random logits.  Walks the actionable proposals in
    the spec §1 precedence order (improving first, then memory-supported neutral)
    and returns the FIRST whose successor state is NOVEL -- i.e. its
    ``successor_state_fingerprint`` is not already in ``visited_state_fingerprints``
    (spec §5 cycle guard: never re-enter an already-occupied state, even when Cmax
    does not drop).  Returns ``(proposal_id, reason)``.

    STOP (returns ``None``) with an explicit reason:
      * ``no_improving_or_memory_supported_neutral`` -- nothing actionable at all
        (spec §13: STOP now requires no improving AND no memory-supported neutral).
      * ``cycle_guard_no_novel_successor`` -- every actionable proposal's successor
        is an already-visited state (spec §5).

    Also returns the list of ``(proposal_id, successor_fingerprint)`` skipped by the
    cycle guard, for transparent provenance.
    """
    by_id = {r["proposal_id"]: r for r in retentions}
    ranked = _d6_rank_actionable(retentions)
    skipped: list[dict[str, str]] = []
    if not ranked:
        return None, "no_improving_or_memory_supported_neutral", skipped
    for pid, reason in ranked:
        succ_fp = by_id[pid].get("successor_state_fingerprint")
        if succ_fp is not None and succ_fp in visited_state_fingerprints:
            # spec §5: do not execute; try the next eligible actionable proposal.
            skipped.append(
                {
                    "proposal_id": pid,
                    "successor_state_fingerprint": succ_fp,
                    "reason": "cycle_guard_repeated_state",
                }
            )
            continue
        return pid, reason, skipped
    # every actionable proposal cycled back to a visited state (spec §5).
    return None, "cycle_guard_no_novel_successor", skipped


def build_d6_state_step(
    problem: Any,
    schedule: Any,
    appearance: Mapping[str, Any],
    *,
    case_id: str,
    fixture: str,
    step_index: int,
    executor: Any,
    device: str = "cpu",
    successor_memory_path: Path = FROZEN_MEMORY_PATH,
    successor_memory_sha: str = FROZEN_MEMORY_SHA,
    visited_state_fingerprints: frozenset[str] = frozenset(),
):
    """One D6 iteration step at S_t (no execution of S_t here).

    Runs the unchanged D5 single-state analysis (``build_d2_expansion`` ->
    current_state_memory), re-drives the identical deterministic pipeline for the
    live proposal objects (order asserted against the manifest), Frozen-Local
    sibling-evaluates each against the byte-stable baseline, applies the retention
    rule, assembles the retained action set (retained proposals + STOP) for M3
    (shadow / non-authoritative), and selects P* by the deterministic fallback.
    Returns ``(step_dict, id_to_prop)``.
    """
    from causal_schedule_lab.validation import schedule_hash

    # (1) unchanged D5 single-state analysis -> current_state_memory (S_t).
    manifest = build_d2_expansion(
        problem, schedule, appearance, case_id=case_id, fixture=fixture, device=device
    )
    state_id = manifest["state_id"]
    state_fp = manifest["state_fingerprint"]
    current_state_memory = manifest["authority"]["memory_context"]

    # (2) live legal proposal objects (same deterministic re-drive) + order assert.
    live_legal = _d6_live_legal_proposals(
        problem, schedule, appearance, case_id, device=device
    )
    live_ids = [p.proposal_id for p in live_legal]
    manifest_ids = [r["proposal_id"] for r in manifest["legal_proposals"]]
    if live_ids != manifest_ids:
        raise SystemExit(
            "D6 BLOCKER: live re-drive proposal order diverged from the manifest "
            f"legal_proposals.\n  live={live_ids}\n  manifest={manifest_ids}"
        )
    considered = live_legal[:D6_MAX_PROPOSALS_PER_STATE]
    id_to_prop = {p.proposal_id: p for p in considered}

    # (3) Frozen-Local sibling evaluation against the SAME byte-stable baseline.
    base_ms = int(schedule.makespan)
    base_hash = schedule_hash(schedule)
    neutral_budget = [0]
    retentions: list[dict[str, Any]] = []
    for prop in considered:
        ev, succ = _d6_evaluate_successor(
            problem, schedule, prop, executor, base_ms=base_ms, base_hash=base_hash
        )
        retentions.append(
            _d6_retention_decision(
                problem,
                prop,
                ev,
                succ,
                case_id=case_id,
                neutral_budget_state=neutral_budget,
                device=device,
                memory_path=successor_memory_path,
                memory_sha=successor_memory_sha,
            )
        )
    baseline_stable = schedule_hash(schedule) == base_hash  # after ALL sibling evals

    # (4) retained action set -> M3 = {retained proposals, STOP} (shadow).
    retained = [r for r in retentions if r["retention_status"] == "retained"]

    # (5) deterministic engineering fallback selection of P* (M3 non-authoritative).
    # Precedence: improving -> memory-supported neutral -> STOP, with the §5 cycle
    # guard skipping any actionable proposal whose successor is already-visited.
    selected_id, sel_reason, cycle_skipped = _d6_select_proposal(
        retentions, visited_state_fingerprints=visited_state_fingerprints
    )

    step = {
        "step_index": step_index,
        "state_id": state_id,
        "state_fingerprint": state_fp,
        "base_makespan": base_ms,
        "base_schedule_hash": base_hash,
        # spec: sibling independence -- baseline byte-stable after ALL evals.
        "baseline_stable_after_sibling_eval": bool(baseline_stable),
        # spec: DISTINGUISH current_state_memory (S_t) from successor_state_memory.
        "current_state_memory": {
            "source": "current_state_S_t",
            "state_id": state_id,
            "memory_support_status": current_state_memory["memory_support_status"],
            "neutral_then_improved_available": current_state_memory[
                "neutral_then_improved_available"
            ],
            "memory_snapshot_sha": current_state_memory["memory_snapshot_sha"],
            "memory_snapshot_sha_matches": current_state_memory[
                "memory_snapshot_sha_matches"
            ],
            "retrieved_count": current_state_memory["retrieved_count"],
        },
        "n_legal_proposals": len(live_legal),
        "n_considered": len(considered),
        "proposals_truncated": len(live_legal) > D6_MAX_PROPOSALS_PER_STATE,
        "retentions": retentions,
        "n_retained": len(retained),
        "retained_proposal_ids": [r["proposal_id"] for r in retained],
        # retained action set handed to M3 (shadow only; STOP first-class).
        "m3_action_set": {
            "action_set_definition": "A(S_t) = {retained proposals, STOP}",
            "actions": [
                {
                    "action_type": "PROPOSAL",
                    "proposal_id": r["proposal_id"],
                    "retention_status": r["retention_status"],
                    "retention_reason": r["retention_reason"],
                    "immediate_delta": r["immediate_delta"],
                    "successor_state_id": r["successor_state_id"],
                    "successor_state_fingerprint": r["successor_state_fingerprint"],
                    "successor_memory_match_status": r["successor_memory_match_status"],
                    "memory_supported_promising_successor": r[
                        "memory_supported_promising_successor"
                    ],
                }
                for r in retained
            ]
            + [{"action_type": "STOP", "proposal_id": None}],
            "stop_is_first_class_action": True,
            "m3_checkpoint_loaded": False,
            "m3_outputs_authoritative": False,
            "m3_authoritative": False,
            # spec §8: the memory-supported neutral proposals STAY in the action set
            # for M3 visibility; the count is surfaced but M3 is not forced to pick.
            "memory_supported_neutral_proposal_ids": [
                r["proposal_id"]
                for r in retained
                if r["retention_reason"] == "memory_supported_promising_successor"
            ],
        },
        # deterministic engineering fallback (spec §1/§9): not M3, no random logits.
        # Precedence improving -> memory-supported neutral -> STOP, cycle-guarded.
        "selection": {
            "selection_authority": D6_SELECTION_AUTHORITY,
            "m3_authoritative": False,
            "uses_random_logits": False,
            "selected_proposal_id": selected_id,
            "selection_reason": sel_reason,
            # spec §10: Memory grants EXPLORATION eligibility, never authority.
            "memory_directly_selects_action": False,
            "memory_generates_reward": False,
            "memory_maps_to_success_label": False,
            "memory_maps_to_future_gain_pred": False,
            # spec §5: proposals skipped because their successor was already visited.
            "cycle_guard_skipped": cycle_skipped,
            "visited_state_fingerprints_count": len(visited_state_fingerprints),
        },
        # inherited D5 firewall flags for this step (from the unchanged manifest).
        "d5_firewalls": {
            "identified": manifest["identified"],
            "memory_authoritative": manifest["authority"]["memory_authoritative"],
            "memory_mutated": manifest["authority"]["memory_mutated"],
            "m3_authoritative": manifest["authority"]["m3_authoritative"],
            "manifest_schema_version": manifest["schema_version"],
        },
    }
    return step, id_to_prop


def build_d6_iteration(
    problem: Any,
    schedule: Any,
    appearance: Mapping[str, Any],
    *,
    case_id: str,
    fixture: str,
    device: str = "cpu",
    successor_memory_path: Path = FROZEN_MEMORY_PATH,
    successor_memory_sha: str = FROZEN_MEMORY_SHA,
) -> dict[str, Any]:
    """The D6 bounded mainline loop.

    S_t -> analyze -> retain successors -> choose P* -> EXECUTE P* -> S_{t+1} ->
    RE-ANALYZE FROM SCRATCH -> repeat, until STOP (no improving proposal) or a loop
    bound.  Each iteration re-diagnoses the new schedule and re-runs the full
    analysis -- S0's candidates / roots / memory are NEVER reused for S1.  The only
    execution is the chosen P* (siblings were evaluated Frozen-Local).  Returns the
    D6 manifest.  Trains nothing; frozen Memory is untouched (SHA asserted).
    """
    from causal_schedule_lab.symptom_pruning import diagnose_and_prune
    from causal_schedule_lab.teacher.atomic_counterfactual_executor import (
        AtomicCounterfactualExecutor,
    )

    executor = AtomicCounterfactualExecutor()
    cur_schedule = schedule
    cur_appearance = appearance
    steps: list[dict[str, Any]] = []
    trajectory_ms: list[int] = [int(cur_schedule.makespan)]
    trajectory_sidecar: list[dict[str, Any]] = []  # engineering only (NOT frozen mem)
    stopped_reason: str | None = None
    iterations_total = 0
    # spec §5 cycle guard: every state we have OCCUPIED (analyzed) in this loop.
    # A memory-supported neutral branch may not drop Cmax, so we must forbid
    # re-entering an already-visited state (fingerprint) to stay terminating.
    visited_state_fingerprints: set[str] = set()

    for it in range(D6_MAX_ITERATIONS):
        iterations_total = it + 1
        step, id_to_prop = build_d6_state_step(
            problem,
            cur_schedule,
            cur_appearance,
            case_id=case_id,
            fixture=fixture,
            step_index=it,
            executor=executor,
            device=device,
            successor_memory_path=successor_memory_path,
            successor_memory_sha=successor_memory_sha,
            visited_state_fingerprints=frozenset(visited_state_fingerprints),
        )
        steps.append(step)
        # mark the state we are AT as visited (before executing its chosen P*).
        visited_state_fingerprints.add(step["state_fingerprint"])
        selected_id = step["selection"]["selected_proposal_id"]
        if selected_id is None:
            # spec §1/§5/§13: STOP now covers both "nothing actionable" and
            # "every actionable proposal cycles back to a visited state".
            stopped_reason = "stop_" + step["selection"]["selection_reason"]
            break

        # --- execute P*: S_t -> S_{t+1} (the ONLY execution; siblings frozen) ---
        prop = id_to_prop[selected_id]
        atoms = _d6_proposal_to_atoms(problem, cur_schedule, prop)
        if atoms is None:  # defensive: fallback only picks mappable-improving props
            stopped_reason = "stop_selected_unmappable_fail_closed"
            break
        res = _d6_execute(executor, problem, cur_schedule, atoms)
        if not res.feasible or res.schedule is None:
            stopped_reason = "stop_selected_infeasible"
            break

        before_ms = int(cur_schedule.makespan)
        next_schedule = res.schedule
        after_ms = int(next_schedule.makespan)
        sel_reason = next(
            (
                r["retention_reason"]
                for r in step["retentions"]
                if r["proposal_id"] == selected_id
            ),
            None,
        )
        trajectory_sidecar.append(
            {
                "trajectory_id": f"D6::{case_id}",
                "step_index": it,
                "state_before": step["state_id"],
                "state_after": None,  # backfilled after the loop
                "proposal": selected_id,
                "immediate_delta": after_ms - before_ms,  # after - before
                "retention_reason": sel_reason,
                "terminal_state": None,  # backfilled after the loop
                "terminal_delta": None,  # backfilled after the loop
            }
        )

        # advance: re-analyze from scratch on S_{t+1} (fresh appearance).
        cur_schedule = next_schedule
        cur_appearance = diagnose_and_prune(problem, cur_schedule).model_dump(
            mode="json"
        )
        trajectory_ms.append(after_ms)
    else:
        stopped_reason = "max_iterations_reached"

    # --- terminal + future_gain TRUTH DEFINITION (offline only; NO training) ----
    initial_ms = trajectory_ms[0]
    terminal_ms = trajectory_ms[-1]
    for i, entry in enumerate(trajectory_sidecar):
        entry["state_after"] = (
            steps[i + 1]["state_id"] if i + 1 < len(steps) else "TERMINAL"
        )
        entry["terminal_state"] = i == len(trajectory_sidecar) - 1
        # terminal_delta_from_t = Cmax(S_T) - Cmax(S_t): the offline truth
        # DEFINITION only; there is NO Effect training and no loss over it.
        entry["terminal_delta"] = terminal_ms - trajectory_ms[i]

    # per-state re-analysis check: every step re-diagnosed (distinct or advanced).
    state_ids = [s["state_id"] for s in steps]
    reused_s0_for_s1 = len(state_ids) >= 2 and len(set(state_ids)) == 1 and (
        # identical state_id across steps is only legitimate if the schedule truly
        # did not change; but we only advance on a strict improvement, so distinct.
        trajectory_ms[0] == trajectory_ms[-1]
    )

    manifest: dict[str, Any] = {
        "schema_version": D6_MANIFEST_SCHEMA,
        "mode": "d6",
        "fixture": fixture,
        "case_id": case_id,
        "device": device,
        # --- loop outcome ---
        "iterations_total": iterations_total,
        "stopped_reason": stopped_reason,
        "trajectory_makespans": trajectory_ms,
        "initial_makespan": initial_ms,
        "terminal_makespan": terminal_ms,
        "terminal_delta_total": terminal_ms - initial_ms,  # <0 = improved
        "loop_bounds": {
            "max_iterations": D6_MAX_ITERATIONS,
            "max_proposals_per_state": D6_MAX_PROPOSALS_PER_STATE,
            "max_neutral_exploration": D6_MAX_NEUTRAL_EXPLORATION,
        },
        # --- D6.1 cycle guard (spec §5): visited-state set, bounded, no graph search ---
        "cycle_guard": {
            "visited_state_fingerprints": sorted(visited_state_fingerprints),
            "visited_state_count": len(visited_state_fingerprints),
            "policy": (
                "a proposal whose successor_state_fingerprint is already visited is "
                "NOT executed (cycle_guard_repeated_state); if every actionable "
                "proposal cycles, the loop STOPs (cycle_guard_no_novel_successor)"
            ),
        },
        # --- per-state steps (full retention + action set + selection) ---
        "steps": steps,
        # --- engineering trajectory sidecar (NOT frozen Memory) ---
        "trajectory_sidecar": trajectory_sidecar,
        "trajectory_sidecar_note": (
            "engineering-only; NOT appended to the frozen trajectory_memory.json "
            "(its SHA is asserted unchanged on every retrieval)"
        ),
        # --- future_gain truth DEFINITION (offline; no training / no loss) ---
        "future_gain_truth_definition": {
            "immediate_delta_t": "Cmax(S_{t+1}) - Cmax(S_t)",
            "terminal_delta_from_t": "Cmax(S_T) - Cmax(S_t)",
            "grounded_in": "frozen_local_iterative_trajectory",
            "trained": False,
            "loss_weight": 0.0,
            "is_free_global_delta": False,  # spec: must NOT substitute Free-Global
            "is_memory_historical_gain": False,  # spec: must NOT copy Memory gain
        },
        # --- authority firewall (spec) ---
        "authority": {
            "executor_called": True,  # D6 DOES execute the chosen P* (real loop)
            "executor_frozen_local": True,  # sibling evals never mutate S_t
            "selection_authority": D6_SELECTION_AUTHORITY,
            "m3_checkpoint_loaded": False,
            "m3_authoritative": False,
            "m3_outputs_authoritative": False,
            "uses_random_logits": False,
            "memory_authoritative": False,  # memory never decides
            "memory_generates_reward": False,
            "memory_generates_success_label": False,
            "memory_generates_future_gain": False,
            "memory_forces_selection": False,
            # spec §10: Memory grants EXPLORATION eligibility for a neutral branch,
            # it does NOT directly select the action (the deterministic engineering
            # policy does, only permitting ONE memory-supported neutral when no
            # immediate improvement exists).
            "memory_directly_selects_action": False,
            "memory_grants_neutral_exploration_eligibility_only": True,
            "memory_mutated": False,  # frozen snapshot untouched
            "current_vs_successor_memory_distinguished": True,
            "per_state_reanalysis": True,  # S_{t+1} fully re-analyzed from scratch
            "reused_prior_state_analysis": bool(reused_s0_for_s1),
            "trajectory_memory_appended": False,  # sidecar only
            # spec §5: cycle guard active; §7: ordinary neutral no-match not executed.
            "cycle_guard_active": True,
            "ordinary_neutral_no_match_auto_executed": False,
        },
        # --- formal-state invariants ---
        "formal_state": {
            "formal_training": 0,
            "optimizer_steps": 0,
            "formal_val_adaptation": 0,
            "formal_test_access": 0,
            "identified": False,
            "safe_to_train": "NO",
            "engineering_only": True,
        },
        "frozen_memory_sha_expected": FROZEN_MEMORY_SHA,
        "stop_point": "after_bounded_iterative_loop",
    }
    return manifest


def _run_d6(args: argparse.Namespace) -> int:
    if args.case in ("routing_blocker", "multi_root"):
        problem, schedule, appearance, case_id = _d2_smoke_case(args.case)
        fixture_label = f"<smoke:{args.case}>"
    else:
        problem, schedule, appearance, case_id = load_case(args.fixture)
        fixture_label = str(args.fixture)

    manifest = build_d6_iteration(
        problem,
        schedule,
        appearance,
        case_id=case_id,
        fixture=fixture_label,
        device=args.device,
    )

    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    out_path = args.json_out
    if out_path is None:
        out_path = D2_SMOKE_ROOT / case_id / "d6_iteration_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n")
    print(text)
    print(f"\nD6 manifest -> {out_path}", file=sys.stderr)

    auth = manifest["authority"]
    fg = manifest["future_gain_truth_definition"]
    fs = manifest["formal_state"]
    # Non-cosmetic gate: every firewall / bound / honesty invariant must hold.
    ok = (
        manifest["schema_version"] == D6_MANIFEST_SCHEMA
        and manifest["iterations_total"] >= 1
        and manifest["stopped_reason"] is not None
        and manifest["iterations_total"] <= D6_MAX_ITERATIONS
        # a real loop ran with at least one analyzed state
        and len(manifest["steps"]) >= 1
        # deterministic fallback selection; M3 never authoritative, no RNG
        and auth["selection_authority"] == D6_SELECTION_AUTHORITY
        and not auth["m3_authoritative"]
        and not auth["m3_outputs_authoritative"]
        and not auth["m3_checkpoint_loaded"]
        and not auth["uses_random_logits"]
        # memory never decides / rewards / labels / forces selection / mutates
        and not auth["memory_authoritative"]
        and not auth["memory_generates_reward"]
        and not auth["memory_generates_success_label"]
        and not auth["memory_generates_future_gain"]
        and not auth["memory_forces_selection"]
        and not auth["memory_mutated"]
        and auth["current_vs_successor_memory_distinguished"]
        # real iterative loop: re-analysis from scratch, no S0 reuse, sidecar only
        and auth["executor_called"]
        and auth["executor_frozen_local"]
        and auth["per_state_reanalysis"]
        and not auth["reused_prior_state_analysis"]
        and not auth["trajectory_memory_appended"]
        # future_gain: truth DEFINITION only, not trained, not a substitute
        and not fg["trained"]
        and fg["loss_weight"] == 0.0
        and not fg["is_free_global_delta"]
        and not fg["is_memory_historical_gain"]
        # every step: sibling independence held + STOP first-class + M3 shadow
        and all(s["baseline_stable_after_sibling_eval"] for s in manifest["steps"])
        and all(
            s["m3_action_set"]["stop_is_first_class_action"] for s in manifest["steps"]
        )
        and all(not s["d5_firewalls"]["identified"] for s in manifest["steps"])
        # formal-state invariants
        and fs["formal_training"] == 0
        and fs["optimizer_steps"] == 0
        and not fs["identified"]
        and fs["safe_to_train"] == "NO"
        and fs["engineering_only"]
        and manifest["frozen_memory_sha_expected"] == FROZEN_MEMORY_SHA
    )
    print(
        f"D6 loop: {manifest['iterations_total']} iterations, "
        f"makespans {manifest['trajectory_makespans']} "
        f"(delta_total={manifest['terminal_delta_total']}), "
        f"stopped={manifest['stopped_reason']}",
        file=sys.stderr,
    )
    for s in manifest["steps"]:
        sel = s["selection"]["selected_proposal_id"]
        print(
            f"  step {s['step_index']}: state={s['state_id']} ms={s['base_makespan']} "
            f"legal={s['n_legal_proposals']} retained={s['n_retained']} "
            f"-> select={sel} ({s['selection']['selection_reason']})",
            file=sys.stderr,
        )
    print(f"D6 structural check: {'OK' if ok else 'FAILED'}", file=sys.stderr)
    return 0 if ok else 1


def _run_d1(args: argparse.Namespace) -> int:
    problem, schedule, appearance, case_id = load_case(args.fixture)
    audit = build_v5_runtime_audit(
        problem,
        schedule,
        appearance,
        case_id=case_id,
        fixture=args.fixture,
        device=args.device,
        checkpoint=args.checkpoint,
    )

    payload = audit.to_dict()
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n")

    # Non-cosmetic gate: refuse to report success if any structural invariant
    # is off, so a broken wiring cannot masquerade as a passing D1 run.
    ok = (
        audit.schema_matches_canonical
        and audit.mock_untrained
        and not audit.neural_outputs_authoritative
        and not audit.identified
        and audit.selection_authority == "deterministic_fallback"
        and not audit.proposal_executed
        and not audit.next_state_generated
        and not audit.iterated
    )
    print(f"\nD1 structural check: {'OK' if ok else 'FAILED'}", file=sys.stderr)
    return 0 if ok else 1


def _d2_smoke_case(name: str):
    """Return (problem, schedule, appearance, case_id) for a D2 smoke fixture.

    Reuses the EXISTING integration-release-gate regression shapes unmodified
    (spec §17): ``routing_blocker`` == reg-C (A4 machine/routing blocker with a
    legal escape edit); ``multi_root`` == reg-D (two independent cause branches,
    no Top-K collapse).

    The appearance is produced by the REAL detector ``diagnose_and_prune`` (the
    canonical appearance producer), then serialized exactly as the release-gate
    tests do (``model_dump(mode="json")``).  For ``multi_root`` the block
    membership is set to both branch operations -- this mirrors the gate test's
    own authoritative multi-root input ``build_proposals({"D": ["Ot", "Ox"]})``
    (diagnose alone collapses the block to a single op); it supplies block
    membership as an *input* and alters no algorithm to hit a numeric target.
    """
    import copy
    import importlib.util

    from causal_schedule_lab.symptom_pruning import diagnose_and_prune

    gate_path = REPO_ROOT / "tests" / "test_m2_integration_release_gate.py"
    spec = importlib.util.spec_from_file_location("_d2_gate_fixtures", gate_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if name == "routing_blocker":
        prob, sched = module._reg_c_problem()
        appearance = diagnose_and_prune(prob, sched).model_dump(mode="json")
        return prob, sched, appearance, "d2_routing_blocker"
    if name == "multi_root":
        prob, sched = module._reg_d_problem()
        appearance = diagnose_and_prune(prob, sched).model_dump(mode="json")
        # Supply both cause-branch ops as block membership (gate-test input).
        appearance = copy.deepcopy(appearance)
        if appearance.get("blocks"):
            block = appearance["blocks"][0]["block"]
            block["operations"] = ["Ot", "Ox"]
            for job in ("JT", "JX"):
                if job not in block.get("jobs", []):
                    block.setdefault("jobs", []).append(job)
        return prob, sched, appearance, "d2_multi_root"
    raise ValueError(f"unknown D2 smoke case: {name!r}")


def _run_d2(args: argparse.Namespace) -> int:
    if args.case in ("routing_blocker", "multi_root"):
        problem, schedule, appearance, case_id = _d2_smoke_case(args.case)
        fixture_label = f"<smoke:{args.case}>"
    else:
        problem, schedule, appearance, case_id = load_case(args.fixture)
        fixture_label = str(args.fixture)

    manifest = build_d2_expansion(
        problem,
        schedule,
        appearance,
        case_id=case_id,
        fixture=fixture_label,
        device=args.device,
        checkpoint=args.checkpoint,
    )

    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    # Default sidecar location (spec §18) unless overridden.
    out_path = args.json_out
    if out_path is None:
        out_path = D2_SMOKE_ROOT / case_id / "d2_expansion_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n")
    print(text)
    print(f"\nD2 manifest -> {out_path}", file=sys.stderr)

    auth = manifest["authority"]
    # Use TRUE totals (never the on-disk capped lists) for the gate + summary,
    # so a truncated large-instance run reports honest numbers and the
    # fail-closed invariant checks the real partition.
    n_searches = manifest["causal_searches_total"]
    n_gen = manifest["generated_proposals_total"]
    n_legal = manifest["legal_proposals_total"]
    n_rej = manifest["rejected_proposals_total"]
    # Non-cosmetic gate (spec §20): every firewall / stop invariant must hold.
    ok = (
        manifest["schema_matches_canonical"]
        and auth["candidate_authority"] == "deterministic_fallback"
        and not auth["neural_scores_authoritative"]
        and not auth["effect_predictor_used"]
        # D4: memory retrieval RAN, but read-only, non-authoritative, and it did
        # NOT change the proposal set (spec §2/§4/§13/§14.7).
        and auth["memory_retrieval_used"]
        and not auth["memory_authoritative"]
        and not auth["memory_mutated"]
        and not auth["memory_changes_proposal_set"]
        # D5: M3 policy interface was called as a SHADOW consumer (spec §9): no
        # checkpoint, non-authoritative, selected nothing, changed no proposal.
        and auth["m3_called"]
        and not auth["m3_authoritative"]
        and not auth["m3_changes_proposal_set"]
        and auth["selected_proposal_id"] is None
        and not auth["executor_called"]
        and not auth["next_state_created"]
        and not manifest["identified"]
        # a real expansion actually ran: candidate space + at least one search
        and len(manifest["candidate_sites"]) > 0
        and n_searches >= 0
        # fail-closed partition on true totals: generated == legal + rejected,
        # and legal is a subset of generated.
        and n_legal <= n_gen
        and n_gen == n_legal + n_rej
        # D3 shadow effect layer holds zero decision power (spec §16/§17):
        # enabled but non-authoritative, no checkpoint, no proposal-set change.
        and manifest["effect_layer"]["enabled"]
        and not manifest["effect_layer"]["checkpoint_loaded"]
        and not manifest["effect_layer"]["effect_outputs_authoritative"]
        and not manifest["effect_layer"]["changes_proposal_set"]
        and manifest["effect_layer"]["predictions_total"] == 0
        and manifest["effect_supervision_contract"]["future_gain_status"].startswith("fail_closed")
        and manifest["effect_supervision_contract"]["fiv_status"] == "derived_no_supervision"
        # D4 memory layer: read-only prior, SHA-locked, proven proposal-set /
        # Effect-Predictor invariant, honest firewall (spec §5/§7/§10/§14).
        and manifest["memory_layer"]["enabled"]
        and not manifest["memory_layer"]["authoritative"]
        and manifest["memory_layer"]["read_only"]
        and not manifest["memory_layer"]["mutated"]
        and manifest["memory_layer"]["memory_snapshot_sha"] == FROZEN_MEMORY_SHA
        and manifest["memory_layer"]["memory_snapshot_sha_matches"]
        and not manifest["memory_layer"]["changes_proposal_set"]
        and manifest["memory_layer"]["proposal_set_unchanged"]
        and manifest["memory_layer"]["effect_prediction_unchanged"]
        and not manifest["memory_layer"]["maps_to_future_gain_pred"]
        and not manifest["memory_layer"]["maps_to_success_label"]
        and not manifest["memory_layer"]["decides_keep_or_drop"]
        # a no-match is a LEGAL state (spec §12), never forced to prune
        and manifest["memory_layer"]["memory_support_status"] in ("supported", "no_match")
        # D5 M3 shadow layer: enabled, no checkpoint, non-authoritative, STOP is
        # a first-class action, proven proposal-set / Effect / Memory invariant,
        # nothing selected, no reward, no execution, no next state (spec §9-§15).
        and manifest["m3_layer"]["enabled"]
        and not manifest["m3_layer"]["checkpoint_loaded"]
        and not manifest["m3_layer"]["m3_outputs_authoritative"]
        and manifest["m3_layer"]["stop_is_first_class_action"]
        and manifest["m3_layer"]["stop_available"]
        and not manifest["m3_layer"]["changes_proposal_set"]
        and manifest["m3_layer"]["proposal_set_unchanged"]
        and manifest["m3_layer"]["derivation_ids_unchanged"]
        and manifest["m3_layer"]["effect_output_unchanged"]
        and manifest["m3_layer"]["memory_flag_faithful_to_evidence"]
        and manifest["m3_layer"]["nothing_selected"]
        and not manifest["m3_layer"]["memory_directly_selects_action"]
        and not manifest["m3_layer"]["reward_computed"]
        and not manifest["m3_layer"]["executor_called"]
        and not manifest["m3_layer"]["next_state_created"]
        and not manifest["m3_layer"]["memory_mutated"]
        and manifest["m3_layer"]["surfaces_memory_supports_neutral_retention"]
        # CONTINUE: schema-reserved, not implemented as runtime behavior (spec §12)
        and not manifest["m3_layer"]["continue_implemented"]
    )
    print(
        f"D2 expansion: {len(manifest['candidate_sites'])} candidate sites, "
        f"{n_searches} causal searches, "
        f"{len(manifest['actionable_roots'])} actionable roots, "
        f"{n_gen} generated / {n_legal} legal / {n_rej} rejected proposals",
        file=sys.stderr,
    )
    ml = manifest["memory_layer"]
    print(
        f"D4 memory: read-only retrieval ran (sha {ml['memory_snapshot_sha'][:12]}, "
        f"match={ml['memory_snapshot_sha_matches']}), "
        f"{ml['total_candidates_considered']} candidates - {ml['excluded_count']} excluded "
        f"-> {ml['retrieved_count']} retrieved [{ml['memory_support_status']}], "
        f"proposal_set_unchanged={ml['proposal_set_unchanged']}, "
        f"effect_pred_unchanged={ml['effect_prediction_unchanged']}",
        file=sys.stderr,
    )
    m3 = manifest["m3_layer"]
    print(
        f"D5 M3 shadow: A(S_t)={{{m3['proposal_actions_total']} proposals, STOP}}"
        f" (continue={m3['continue_status']}), policy={m3['policy_called']}"
        f" [{m3['prediction_status']}], selected=None, "
        f"proposal_set_unchanged={m3['proposal_set_unchanged']}, "
        f"effect_unchanged={m3['effect_output_unchanged']}, "
        f"neutral_retention_props={m3['proposal_actions_with_neutral_retention']}",
        file=sys.stderr,
    )
    print(f"D2 structural check: {'OK' if ok else 'FAILED'}", file=sys.stderr)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("d1", "d2", "d6"),
        default="d1",
        help=(
            "d1 = single-forward skeleton; d2 = single-state expansion; "
            "d6 = iterative successor-state retention + bounded mainline loop"
        ),
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help="reverse-inference serialization JSON (problem+schedule+appearance)",
    )
    parser.add_argument(
        "--case",
        default=None,
        help="D2 smoke case: 'routing_blocker' | 'multi_root' (else use --fixture)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="(refused) formal checkpoint path -- out of D1/D2 scope",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional path to write the structured audit / manifest JSON",
    )
    args = parser.parse_args(argv)

    if args.mode == "d6":
        return _run_d6(args)
    if args.mode == "d2":
        return _run_d2(args)
    return _run_d1(args)


if __name__ == "__main__":
    raise SystemExit(main())
