"""Canonical M3 training config.

Values are VERBATIM from the legacy R-chain (`scripts/t1_m3_canonical_agentic_rl_r1.py` == r1,
`scripts/t1_m3_proposal_sft_ranking_r2.py` == r2, `scripts/t1_m3_ranking_and_progressive_memory_r4.py` == r4).
Single source of truth for the canonical pipeline -- do not redefine numbers locally.
"""

import os
from pathlib import Path

# -- project layout ----------------------------------------------------------
# this file lives at src/causal_schedule_lab/m3/config.py
ROOT = Path(__file__).resolve().parents[3]

# canonical outputs live here (consolidation Phase 0)
CANONICAL_OUT_DIR = ROOT / "outputs" / "canonical_m3"
# canonical checkpoints
SFT_CKPT = CANONICAL_OUT_DIR / "m3_proposal_sft_v1.pt"
GRPO_CKPT = CANONICAL_OUT_DIR / "m3_grpo_v1.pt"
# upstream frozen assets (shared; never move / never archive)
UTIL_DIR = ROOT / "outputs" / "m3_makespan_utility"
DEFAULT_CKPT = ROOT / "outputs" / "b5_1_pilot_A1_v1_1_adapter" / "b5_1_shared.pt"

# -- proposal feature space (r1) ---------------------------------------------
HORIZON = 5
PAIR_TOP_K = 100
B5_EPS = 1e-3
SINGLE_FEAT_DIM = 12
PAIR_FEAT_DIM = 16
PROP_FEAT_DIM = 300          # [h_u(128) h_v(128) feat_u(12) feat_v(12) pstruct(16)
                             #  uhat_u(1) uhat_v(1) direct(1) is_pair(1)]
STATE_FEAT_DIM = 7

# -- SFT scorer (r2) ----------------------------------------------------------
MEM_FEAT_DIM = 6
SCORER_HIDDEN = 256
GATE_HIDDEN = 64
LAMBDA_CLS = 1.0             # fused score = rank + LAMBDA_CLS * logit(P(U>0))
EPOCHS = 15
LR = 1e-3
RANK_PAIRS_PER_STATE = 64
INFEASIBLE_U = -1e9          # ranking sentinel for cycle/infeasible proposals
USE_REG_HEAD = False         # regression head optional, not primary

# -- gate (r3) ----------------------------------------------------------------
K_R = 10                     # rank_head top-K
K_C = 10                     # usefulness_head top-K
GATE_EXTRA_DIM = 8           # pooled proposal statistics
GATE_FEAT_DIM = STATE_FEAT_DIM + GATE_EXTRA_DIM     # 15

# -- progressive memory (r4) ---------------------------------------------------
MEM_TOP_N_STATES = 3         # nearest memory states allowed past the gate
MEM_TAU_STD = 2.0            # state-distance cap in standardized (z) space
MEM_TAU_MIN = 0.30           # floor: also require raw z-distance small enough

# -- wide recall + reranker (r4) ------------------------------------------------
WIDE_K_CLS = 30              # Stage-1 usefulness top-K (fixed, no sweep)
WIDE_K_RANK = 20             # Stage-1 rank top-K (fixed)
FINAL_TOP_K = 10
RERANK_HIDDEN = 128
RERANK_EPOCHS = 15
RERANK_LR = 1e-3
MARGIN_NORMAL = 1.0
MARGIN_HARD = 2.0
MISSED_OVERSAMPLE = 4
RERANK_PAIRS_PER_STATE = 200
ROLE_N = 5                   # contributor/enabler/CC/EE/CE

# -- canonical utility-aware reranker (Phase 1) ---------------------------------
UA_UTILITY_SCALE = 20.0      # |U_i-U_j| / utility_scale
UA_MIN_W = 0.2               # clip floor on pair weight
UA_MAX_W = 5.0               # clip ceiling on pair weight
UA_MAX_CATEGORY_FRAC = 0.50  # no category > 50% of sampled pairs
UA_PAIRS_PER_STATE = 160     # Phase-1 reranker pair budget per state

# -- canonical Top-1 utility selection SFT (R6) ---------------------------------
# R6: state-wise listwise top-1.  Group key = (instance_id, state_hash).  Action
# set = WIDE pool ∪ STOP (first-class).  best_idx = argmax true_U in pool if its
# max > 0, else STOP.  Primary objective = CrossEntropy over the M+1 logits, so
# the top-1 (not Pearson) is the decision criterion.  Balanced per state.
TO1_EPOCHS = 15              # SFT epochs (matches RERANK_EPOCHS cadence)
TO1_LR = 1e-3
TO1_PROP_HIDDEN = 128        # selector proposal head (same arch as R5 reranker.net)
TO1_STOP_HIDDEN = 64         # STOP head width
TO1_LAMBDA_PAIR = 0.3        # auxiliary weighted |ΔU| pair loss (fixed v1, no sweep)
TO1_PAIRS_PER_STATE = 64     # pair budget per state for the auxiliary loss
TO1_HARD_OVERSAMPLE = 3      # DPpaulli10a + TOP1_FAILURE_STATE multiplicity (state-balanced)
TO1_STOP_POOL_STAT_DIM = 5   # n_pool + max_old + mean_old + max_rank + max_logit
TO1_CKPT = CANONICAL_OUT_DIR / "m3_proposal_top1_sft_v2.pt"
R6_REPORT = CANONICAL_OUT_DIR / "T1_M3_TOP1_R6_REPORT.md"
# R6 score base = R5 canonical per-proposal representation (frozen scorer hidden
# h + rank + usefulness + old utility + role-oh + state_feat + mem6).  The prop
# head is warm-started from the R5 utility reranker (initialization only, R6 §8);
# the STOP head is fresh and first-class (no gate authority).

# -- canonical R7 STOP calibration + OOD generalization ---------------------------
# R7: fix STOP-vs-Proposal relative scale via margin supervision (first-class
# STOP, no runtime true_U), TRAIN-only state augmentation, memory channel dropout.
# First version: margins m_act = m_stop = 1.0 fixed (no sweep), lambda_stop fixed
# mild, dropout p_drop = 0.5 fixed.  GRPO still forbidden this round.
TO1_MARGIN_ACT = 1.0           # postive: score(best_idx) >= score(STOP) + m_act
TO1_MARGIN_STOP = 1.0          # STOP:   score(STOP) >= max_i score(P_i) + m_stop
TO1_LAMBDA_STOP = 1.0          # L_total = L_top1 + λ_pair·L_pair + λ_stop·L_margin
TO1_MEM_DROP_P = 0.5           # selector mem6 channel dropout (fixed, no sweep)
TO1_EPOCHS_A = 10              # Phase A: proposal backbone/head FROZEN, STOP head only
TO1_EPOCHS_B = 6               # Phase B: last prop layer small-LR joint tune (if A short)
TO1_LR_PHASE_B = 2e-4          # small LR for Phase B (last layer, bounded)
TO1_AUG_PER_INST = 8           # per-instance NEW-unique TRAIN state budget (auto-close)
TO1_AUG_EXPLORE_BASES = 2      # policy states used as exploration bases per instance
TO1_AUG_EXPLORE_K = 2          # legal feasible successors per base (bounded)
TO1_AUG_QUICK_PER_INST = 2     # --quick per-instance budget (smoke only)
TO1_CKPT_R7 = CANONICAL_OUT_DIR / "m3_proposal_top1_sft_v3.pt"
R7_REPORT = CANONICAL_OUT_DIR / "T1_M3_STOP_CALIBRATION_R7_REPORT.md"
# R7 memory semantics unchanged (progressive_causal_time); augmented-state memory
# = fresh per-instance ProgressiveMemory (controlled causal evidence, no future,
# no unseen dump).  true_U remains TRAIN-label / offline-diagnostic ONLY -- never
# a runtime rule (R7 §1).

# -- canonical R8 pool-argmax + cross-instance generalization ---------------------
# R8 (T1-M3-POOL-ARGMAX-AND-CROSS-INSTANCE-GENERALIZATION): primary objective =
# best-vs-hardest-competitor pool-argmax (POOL_ARGMAX), STOP head FROZEN (diagnostic
# only, R8 §2/§8/§36), cross-instance internal validation = fixed 3-fold by INSTANCE
# (all model selection on TRAIN14 held-out instances ONLY; formal VAL run once, no
# tuning, §14-§18).  Runtime still never sees true_U (§3).  GRPO still forbidden (§37).
TO1_R8_ARGMAX_MARGIN = 1.0       # L_argmax = w·relu(score[j_hard] − score[best] + margin)
TO1_R8_LAMBDA_ARGMAX = 1.0       # primary weight (§10); original Top1 CE kept auxiliary
TO1_R8_UTILITY_SCALE = 20.0      # w = clip((U_best−U_j)/scale, TO1_R8_W_MIN, TO1_R8_W_MAX) (§7)
TO1_R8_W_MIN = 0.2
TO1_R8_W_MAX = 5.0
TO1_R8_EPOCHS_A = 12             # Phase A: ONLY top prop-head Linear(128,1), backbone frozen (§9)
TO1_R8_EPOCHS_B = 6              # Phase B: last shared Linear(128,128)+head small LR (§9, gate-only)
TO1_R8_LR_PHASE_A = 1e-3
TO1_R8_LR_PHASE_B = 2e-4
TO1_R8_PHASE_B_GATE_DELTA = 0.05 # Phase B triggers if Phase A positive-state pool-argmax ≤ R6 + delta
TO1_R8_CV_FOLDS = 3              # fixed instance-level folds (no re-rolling, §15-§16)
TO1_R8_CV_SEED = 0
TO1_R8_CV_MIN_IMPROVE = 0.05     # held-out pos-argmax must beat R6 on the SAME held instances by ≥ this
TO1_R8_MEM_GATE = True           # §23-24: memory stays, evidence-only + confidence-gated (g_mem)
TO1_CKPT_R8 = CANONICAL_OUT_DIR / "m3_pool_argmax_sft_v4.pt"
R8_REPORT = CANONICAL_OUT_DIR / "T1_M3_POOL_ARGMAX_R8_REPORT.md"
# R8 keeps: underlying R6 Top-1 selector arc, Wide Recall, M2, Reasoner, pools,
# FixedDecisionReplay, progressive-causal-time Memory.  v4 checkpoint parent =
# m3_proposal_top1_sft_v2.pt (§38).  memory_semantics=progressive_confidence_gated.

# -- canonical R9 auxiliary-instance generalization ------------------------------
# R9 (T1-M3-AUXILIARY-INSTANCE-GENERALIZATION): primary blocker = INSTANCE_DIVERSITY
# / CROSS-INSTANCE GENERALIZATION (R8 proof: L_argmax train-learnable, held-out
# transfer fails).  Policy = R6-anchored supervised RESIDUAL (score_R9 = score_R6 +
# alpha·tanh(delta(P))), delta≡0 init, alpha small — repair R6, never replace (§13).
# Training data = BENCHMARK-TRAIN14 (partial, as real-domain anchor/validation) +
# AUX-TRAIN (synthetic, generated via pre-existing random_fjsp, TRAIN14-derived
# ranges).  Reports keep BENCHMARK-TRAIN14 / AUX-TRAIN / VAL3 / FORMAL-TEST3 separate
# (§4).  Instance-mixed batches (§12), R6-reference preservation L_ref on R6-correct
# states only (§14), improvement-aware supervision (§15).  GRPO still forbidden (§39).
TO1_R9_ALPHA = 0.5              # residual bound: |alpha·tanh(delta)| ≤ alpha (§13)
TO1_R9_LAMBDA_REF = 0.05        # L_ref weight: pin delta≈0 on R6-correct states (§14)
TO1_R9_LAMBDA_ARGMAX = 1.0      # primary weighted pool-argmax loss (§15)
TO1_R9_LR = 2e-3                # residual head LR (small, but res_head is shallow)
TO1_R9_EPOCHS = 8               # full-run epochs (early-stop via held-out epoch sel.)
TO1_R9_CV_EPOCHS = 6            # internal-CV per-fold epochs
TO1_R9_BATCH = 16               # instance-mixed groups per gradient step (§12)
TO1_R9_MIX_BENCH_RATIO = 0.5    # source mixing: 50% benchmark TRAIN / 50% AUX (§12)
TO1_R9_N_AUX = 100              # §6 first-version fixed count (no sweep)
TO1_R9_AUX_HELD_FRAC = 0.2      # §22: AUX train/held split BY INSTANCE
TO1_R9_AUX_SEED = 0
TO1_R9_AUX_HELD_GATE = 0.03     # AUX held-out pos-argmax must beat R6 on same instances by ≥ this
TO1_R9_AUX_DIR = CANONICAL_OUT_DIR.parent / "r9_aux"   # outputs/r9_aux
TO1_AUX_DATA = TO1_R9_AUX_DIR / "r9_aux_data.pt"
TO1_CKPT_R9 = CANONICAL_OUT_DIR / "m3_generalized_sft_v3.pt"
R9_REPORT = CANONICAL_OUT_DIR / "T1_M3_AUXILIARY_INSTANCE_GENERALIZATION_R9_REPORT.md"
AUX_REPORT = CANONICAL_OUT_DIR / "T1_M3_AUXILIARY_INSTANCE_DATA_R9_REPORT.md"

# -- canonical R10 instance-relative score calibration ----------------------------
# R10 (T1-M3-INSTANCE-RELATIVE-SCORE-CALIBRATION): the R6 proposal RANK is FROZEN
# exactly (§2/§17).  Each state's pool is robust within-state normalized
# z_i = (s_i - center)/(scale + eps), center = median, scale = 1.4826*MAD, fallback
# std, then scale=1 (§7) -- parameter-free, state-relative, monotonic positive-affine
# -> deterministic Proposal rank preservation (Spearman=1.0, inversions=0).  A small
# PoolConditionedStopCalibrator (§8-§9) reads policy-observable pool statistics +
# state_feat + old-R6-STOP-raw + g_mem and emits stop_z; final decision =
# argmax(z_1..z_N, stop_z).  STOP only owns ACT-vs-STOP -- never re-ranks Proposals.
# AUX-REAL (D1 SAFE_AUX_REAL real instances) preferred for calibration; AUX synthetic
# as supplement; NOT a data-scaling round (§10-§11).  GRPO still forbidden (§41).
TO1_R10_MAD_CONST = 1.4826               # MAD -> robust scale (§7)
TO1_R10_Z_EPS = 1e-9
TO1_R10_CALIB_HIDDEN = 48                # PoolConditionedStopCalibrator width (small, no sweep)
# calibrator input [18] = state_feat[7] + n_pool + raw{max,median,MAD,std} +
# top1-top2 gap + top1-median + top1 robust-z + top3 mean robust-z + old STOP raw + g_mem
TO1_R10_CALIB_EXTRA_DIM = 11
TO1_R10_CALIB_IN_DIM = STATE_FEAT_DIM + TO1_R10_CALIB_EXTRA_DIM   # 18
TO1_R10_LR = 1e-3
TO1_R10_EPOCHS = 12                      # full-run epochs (CV-selected, held-only)
TO1_R10_CV_EPOCHS = 10                   # internal-CV per-fold epochs
TO1_R10_BATCH = 32                       # class-balanced ACT/STOP micro-batches
TO1_R10_MIX_BENCH_REAL_SYN = (0.5, 0.3, 0.2)  # bench : aux-real : aux-synth (§10-§11)
TO1_R10_REAL_HELD_FRAC = 0.2             # AUX-REAL split BY INSTANCE
TO1_R10_REAL_SEED = 0
TO1_R10_N_REAL = 40                      # AUX-REAL instance count (deterministic, first version)
TO1_R10_MAX_OPS_GATE = 260               # select smallest n_ops <= gate (compute-paced; outliers excluded)
TO1_R10_CV_FOLDS = 3                     # same fixed instance folds as R8/R9
TO1_R10_CV_BAL_ACC = 0.80                # gate: mean held balanced ACT/STOP accuracy
TO1_R10_CV_STOP_RECALL = 0.85            # gate: mean held STOP recall
TO1_R10_CV_FALSE_STOP = 0.15             # gate: mean held false_stop (pos IN pool)
TO1_R10_TRAIN_FALSE_STOP = 0.10          # gate: TRAIN replay false_stop_pos_in_pool
TO1_R10_TRAIN_STOP_RECALL = 0.85         # gate: TRAIN replay STOP recall
TO1_R10_REAL_DIR = CANONICAL_OUT_DIR.parent / "r10_aux_real"   # outputs/r10_aux_real
TO1_REAL_DATA = TO1_R10_REAL_DIR / "r10_aux_real_data.pt"
TO1_CKPT_R10 = CANONICAL_OUT_DIR / "m3_score_calibrated_sft_v3.pt"
R10_REPORT = CANONICAL_OUT_DIR / "T1_M3_SCORE_CALIBRATION_R10_REPORT.md"
R10_AUX_REAL_REPORT = CANONICAL_OUT_DIR / "T1_M3_SCORE_CALIBRATION_R10_AUX_REAL_DATA_REPORT.md"

# -- R11 rolling graph multi-trajectory GRPO (T1-M3-ROLLING-GRAPH-MULTI-TRAJECTORY-GRPO-R11)
# First REAL GRPO round of the canonical line (R1/R2 actor-critic excludes -- directive §0).
# Parent = canonical R6 v2 selector (m3_proposal_top1_sft_v2.pt).  Warm-start bounded
# residual policy: score_GRPO(P) = score_SFT(P) + alpha·tanh(delta(P)) with delta zero-init
# (δ=0 ⇒ policy == R6 behavior; monotone z base ⇒ rank-preserving warm start).  Reward
# R_i = Cmax(S_t) − Cmax(S_T) terminal only; STOP reward = 0.  Group = (iid, state_hash),
# K sibling trajectories, group-relative advantage, clip+KL-to-π_ref, E-epoch batch reuse
# (stale-protected), rolling advancement (no oracle), 40/30/30 bench/real/syn sources,
# parallel trajectory workers.  GRPO now ALLOWED (§0 directive).
TO1_R11_PARENT = TO1_CKPT                     # canonical R6 v2 (m3_proposal_top1_sft_v2.pt)
TO1_R11_ALPHA_PROP = 0.5
TO1_R11_ALPHA_STOP = 0.5
TO1_R11_K = 8                                 # trajectories per (instance_id, state_hash) group
TO1_R11_HORIZON = 5                           # trajectory horizon (STOP may end earlier)
TO1_R11_GRAPHS_PER_BATCH = 4                  # distinct graphs per GRPO batch
TO1_R11_UPDATE_EPOCHS = 3                     # batch reuse epochs (π_old frozen across reuse)
TO1_R11_MAX_ROLLING_STATES_PER_GRAPH = 5      # rolling depth per graph per episode
TO1_R11_TRAINING_CYCLES = 10                  # Stage B fixed cycles (no sweep)
TO1_R11_TEMP = 1.5                            # behavior temperature
TO1_R11_MIX_EPS = 0.10                        # uniform mixture mass (exact mixture logp)
TO1_R11_CLIP_EPS = 0.20                       # PPO clip interval
TO1_R11_BETA_KL = 0.03                        # KL-to-π_ref (frozen R6-base) coefficient
TO1_R11_LR = 1e-3                             # residual-head AdamW LR
TO1_R11_SOURCE_RATIO = (0.4, 0.3, 0.3)        # bench TRAIN14 : AUX-REAL-train : AUX-syn-train
TO1_R11_STALE_KL_THRESHOLD = 1.0              # mean KL(πθ||π_old) above which reuse stops
TO1_R11_STALE_CLIP_FRAC = 0.30                # clip-fraction above which reuse stops
TO1_R11_ADV_EPS = 1e-6
TO1_R11_RESID_SAT = 0.9                       # |tanh δ| > 0.9 counts as saturated
TO1_R11_RESID_SAT_FRAC_LIMIT = 0.50           # >50% saturated residual → collapse guard
TO1_R11_COLLAPSE_TRAIN_RATIO = 0.50           # TRAIN gain < 50% of (same-harness) floor → collapse
TO1_R11_R6_FLOOR = 369                        # R6 B6 recorded TRAIN total (FALLBACK floor; R11 harness
                                              # passes its δ=0 same-harness level as collapse_floor)
TO1_R11_MAX_EPISODES_PER_GRAPH = 3            # graph reuse cap per run (prevent domination)
TO1_R11_STAGE_A_UPDATES = 3                   # Stage A sanity: fixed small iterate count
TO1_CKPT_R11 = CANONICAL_OUT_DIR / "m3_rolling_grpo_v1.pt"
R11_REPORT = CANONICAL_OUT_DIR / "T1_M3_ROLLING_GRPO_R11_REPORT.md"

# -- R12 TRUE-MULTISTEP-ONLINE-ROLLING-GRPO (T1-M3-TRUE-MULTISTEP-ONLINE-ROLLING-GRPO-R12) --
# Parent = m3_rolling_grpo_v1.pt (R11 A).  π_ref stays the R6 SFT reference.  Upgrades:
#   1. FULL multi-step trajectory credit: per-step ratio × SHARED terminal advantage,
#      per-trajectory EQUAL weight (L = mean_i [ mean_k surrogate_i,k ]).
#   2. FORCED CONTINUATION REMOVED: STOP (selector or advancement) truly ends the episode.
#   3. ONLINE state proposal generation: every new S_t re-runs Appearance->M2->pool->wide
#      recall; offline pools only seed / diagnose / held-eval / warm-start.
#   4. trajectory workers = multiprocessing (ProcessPoolExecutor), immutable contract,
#      deterministic seeds == workers=1 .. workers=N.  No ThreadPool as primary path.
TO1_R12_PARENT = TO1_CKPT_R11                        # R11 A checkpoint (m3_rolling_grpo_v1.pt)
TO1_R12_CKPT = CANONICAL_OUT_DIR / "m3_full_trajectory_grpo_v2.pt"
TO1_R12_ALPHA_PROP = 0.5                             # bounded residual envelope (kept from R11)
TO1_R12_ALPHA_STOP = 0.5
TO1_R12_K = 8                                        # trajectories per (iid, state_hash) group
TO1_R12_HORIZON = 5                                  # trajectory horizon (STOP may end earlier)
TO1_R12_GRAPHS_PER_BATCH = 4                         # distinct graphs per GRPO batch (2 bench + 1 real + 1 syn)
TO1_R12_UPDATE_EPOCHS = 3                            # batch reuse (π_old frozen across reuse)
TO1_R12_MAX_DEPTH = 5                                # max rolling depth per graph per episode
TO1_R12_TRAINING_CYCLES = 12                         # Stage B fixed cycles (no sweep)
TO1_R12_TEMP = 1.5                                   # behavior temperature
TO1_R12_MIX_EPS = 0.10                               # uniform mixture mass (exact mixture logp)
TO1_R12_CLIP_EPS = 0.20                              # PPO clip interval
TO1_R12_BETA_KL = 0.03                               # KL-to-π_ref (frozen R6 SFT) coefficient
TO1_R12_LR = 1e-3
TO1_R12_ADV_EPS = 1e-6
TO1_R12_RESID_SAT = 0.9                              # |tanh δ| > 0.9 counts as saturated
TO1_R12_RESID_SAT_FRAC_LIMIT = 0.50
TO1_R12_COLLAPSE_TRAIN_RATIO = 0.50                  # TRAIN gain < 50% of same-ruler floor -> collapse
TO1_R12_MAX_EPISODES_PER_GRAPH = 3                   # graph reuse cap (prevent domination; reset rotates)
TO1_R12_SOURCE_RATIO = (0.4, 0.3, 0.3)               # bench TRAIN14 : AUX-REAL-train : AUX-syn-train
TO1_R12_STALE_KL_THRESHOLD = 1.0
TO1_R12_STALE_CLIP_FRAC = 0.30
TO1_R12_STAGE_A_GRAPHS = 1                           # Stage A correctness run: graphs used
R12_REPORT = CANONICAL_OUT_DIR / "T1_M3_FULL_TRAJECTORY_GRPO_R12_REPORT.md"

# -- R13 JOINT-AGENTIC M2+M3 ROLLING GRPO (T1-M2-M3-JOINT-AGENTIC-ROLLING-GRPO-R13) --
# M2 root-search responsibility ONLY = find roots worth expanding (never the final
# Proposal).  Permanent runtime per step: S_t -> Appearance -> M2 attribution ->
# budgeted root probe (B_ROOT=8: 3 anchor + 4 policy-sampled + 1 uniform outsider)
# -> real direct/dependency probe -> makespan-first filter (Tier A PROVEN_GAIN |
#                                  Tier B MEMORY_RESCUED | Tier C UNSUPPORTED prune)
# -> Reasoner (frozen Contributor+Enabler, dependency-completed) -> complete legal
# Proposal pool -> M3 (R12 semantics) -> execute -> S_{t+1}.
# M2RootPolicyAdapter sits ON the frozen M2 attribution: root_score_RL(u) =
# root_score_M2(u) + alpha_M2 * tanh(delta_root(u)), delta==0 warm start, α_M2 small.
# Memory = second-level evidence only; can NEVER veto a G_probe>0 root.
TO1_R13_PARENT = TO1_CKPT_R11                        # m3_rolling_grpo_v1.pt (R11 A)
TO1_R13_CKPT_M2 = CANONICAL_OUT_DIR / "m2_root_policy_r13.pt"
TO1_R13_CKPT_JOINT = CANONICAL_OUT_DIR / "m2_m3_joint_agentic_grpo_r13.pt"
# per-state probe budget (§12): 3 attribution anchors + 4 policy samples + 1 outsider
TO1_R13_ROOT_BUDGET = 8
TO1_R13_ANCHOR_ROOTS = 3
TO1_R13_POLICY_DRAWS = 4
TO1_R13_OUTSIDER = 1
TO1_R13_MEMORY_BUDGET = 2                            # Tier B cap (§17); Tier A not capped
TO1_R13_ALPHA_M2 = 0.2                               # M2 residual envelope (zero-init, small)
TO1_R13_ALPHA_PROP = 0.5                             # M3 residual kept from R12 (§20, no change)
TO1_R13_ALPHA_STOP = 0.5
TO1_R13_K = 8                                        # trajectories per group
TO1_R13_HORIZON = 5                                  # trajectory horizon (§38, kept)
TO1_R13_GRAPHS_PER_BATCH = 4                         # graphs/batch (≥4; 8 if machine fast) (§37)
TO1_R13_UPDATE_EPOCHS = 3                            # batch reuse E=3 (§39, kept)
TO1_R13_MAX_DEPTH = 5                                # rolling depth per graph (§22-23)
TO1_R13_TRAINING_CYCLES_A = 3                        # Stage A sanity cycles (fixed)
TO1_R13_TRAINING_CYCLES_B = 8                        # Stage B fixed cycles
TO1_R13_TRAINING_CYCLES_C = 10                       # Stage C fixed cycles
TO1_R13_LAMBDA_M2 = 0.5                              # joint loss weight (§28-29, fixed)
TO1_R13_LAMBDA_M3 = 1.0
TO1_R13_BETA_M2 = 0.3                                # M2 KL-to-attribution-prior (beta_M2 > beta_M3)
TO1_R13_BETA_M3 = 0.03                               # M3 KL-to-R6 (R12 value, unchanged)
TO1_R13_TEMP_M2 = 1.5                                # T2-G: flatten M2 root-score gaps
TO1_R13_TEMP = 1.8                                   # T2-G: flatten M3 proposal-score gaps
TO1_R13_MIX_EPS = 0.05                               # legal-action exploration floor
TO1_R13_CLIP_EPS = 0.20                              # PPO clip interval (R12, unchanged)
TO1_R13_LR_M2 = 1e-3                                 # M2 adapter AdamW LR
TO1_R13_LR_M3 = 1e-3                                 # M3 residual AdamW LR (R12)
TO1_R13_M2_FEAT_DIM = 8                              # per-root adapter feature dim
TO1_R13_PROBE_PER_ROOT = 4                           # max proposals executed per root probe
TO1_R13_MEM_SUPPORT_MIN = 2                          # memory rescue: min fine support
TO1_R13_MEM_SUCCESS_MIN = 0.5                        # memory rescue: min fine success rate
R13_REPORT = CANONICAL_OUT_DIR / "T1_M2_M3_JOINT_AGENTIC_GRPO_R13_REPORT.md"

# R13 result citations for the R14 P1/P2 ablation rows (canonical R13 run, 2026-08-28):
# C4 = frozen-M2 x M3-GRPO (gated), C5 = R13 JOINT M2+M3 (shared-terminal credit).
TO1_R13_C4_TRAIN = 353                                 # R13 C4 closed-loop TRAIN (raw ruler)
TO1_R13_C5_TRAIN = 353                                 # R13 C5 JOINT closed-loop TRAIN (raw ruler)
TO1_R13_C5_REAL = 7                                    # R13 C5 AUX-real held improvement (2 -> 7)

# -- R14: T1-M2-M3-STAGEWISE-REWARD-JOINT-GRPO (paper main method, §0-§43) ----------
# Permanent route: M2 SFT (frozen B5 attribution) -> M3 SFT (m3_proposal_top1_sft_v2)
# -> ONE Joint Agentic GRPO stage with STAGEWISE rewards: M2 local q2 (probe) <-> A2
# (§13-15), M3 terminal makespan <-> A3 (§16-17).  NO A/B/C RL training stages (§2);
# M2 adapter + M3 residual both open from cycle 0.  M3 parent is the SFT checkpoint
# ONLY -- m3_rolling_grpo_v1.pt is FORBIDDEN as the final Joint RL parent (§0).
TO1_R14_B_INIT = 8                                   # adaptive probe wave 1 budget (§23)
TO1_R14_B_STEP = 8                                   # expansion wave size
TO1_R14_B_MAX = 24                                   # absolute probe cap (§23-28)
TO1_R14_TIER_A_STOP = 2
TO1_R14_MEM_CONF_SCALE = 2.0                            # memory_confidence(u)=ms/(ms+k) ∈ [0,1); q2_mem=0.5·conf strictly < 0.5 (§10/§34: max(mem)<min(proven) must hold structurally)                              # stop expanding at tier_A >= 2
TO1_R14_EXPAND_POLICY_PER_WAVE = 6                   # expansion: 6 policy + 1 high-attr + 1 outsider
TO1_R14_LAMBDA_M2 = 1.0                              # §22: do NOT pre-suppress M2 (R13 used 0.5)
TO1_R14_LAMBDA_M3 = 1.0
TO1_R14_BETA_M2 = 0.3                                # β2 > β3: protect M2 attribution prior (§21)
TO1_R14_BETA_M3 = 0.03
TO1_R14_TRAINING_CYCLES = 10                         # single JOINT stage (§2), same scale as R13-C
TO1_R14_CKPT = CANONICAL_OUT_DIR / "m2_m3_stagewise_joint_grpo_r14.pt"   # §41, PASS only
R14_REPORT = CANONICAL_OUT_DIR / "T1_M2_M3_STAGEWISE_REWARD_JOINT_GRPO_R14_REPORT.md"

# -- R15: coverage-preserving M3 proposal-selection shortlist (T1-M3-COVERAGE-PRESERVING-SELECTION-GRPO-R15)
# §5-9: K_GLOBAL GLOBAL top-k by frozen M3 SFT base score; per-root quota (Tier-A top-2,
# Tier-B top-1) by frozen score; structural diversity fill (single ROUTE / single SEQ /
# joint ROUTE+ROUTE / joint ROUTE+SEQ / joint SEQ+SEQ); CAP=32; STOP never inside the 32.
TO1_R15_K_GLOBAL = 12                                # §5 Source1 GLOBAL top-k (no sweep)
TO1_R15_TIER_A_QUOTA = 2                             # §6 per retained Tier-A root -> top-2
TO1_R15_TIER_B_QUOTA = 1                             # §7 per retained Tier-B root -> top-1
TO1_R15_SHORTLIST_CAP = 32                           # §8 M3_SHORTLIST_CAP (no sweep); STOP = idx 32
TO1_R15_DIVERSITY_FILL = True                        # §8 structural diversity fill if < CAP
TO1_R15_RECALL_DROP_TOL = 0.05                       # §13 hard gate: 0.05 visible drop -> SHORTLIST_COVERAGE_FAILURE
TO1_R15_SL_SIGNATURE_HASH = "sha256"                 # §20 shortlist signature hash scheme
TO1_R15_TRAINING_CYCLES = 10                         # §16-18 ONE joint stage, same scale as R14
# §25-26 SAME unified RAW ruler: Q0/Q2 are CITATIONS from the R14 run records
# (P0 gated-SFT full pool = 355, P3 stagewise-JOINT full pool = 356); Q1/Q3 computed
# fresh on the R15 SFT parent / final policy with the shortlist action set.
TO1_R15_Q0_CITE_FULL_SFT = 355.0                     # R14 P0 (gated SFT, full pool)
TO1_R15_Q2_CITE_FULL_JOINT = 356.0                   # R14 P3 (stagewise JOINT, full pool)
TO1_R15_MISS_INIT_REF = 3                            # R14 M3_SELECTION_MISS reference (before)
TO1_R15_CKPT = CANONICAL_OUT_DIR / "m2_m3_stagewise_joint_grpo_r15.pt"   # §44, PASS(verdict A) ONLY
R15_REPORT = CANONICAL_OUT_DIR / "T1_M3_COVERAGE_PRESERVING_SELECTION_GRPO_R15_REPORT.md"

# -- R16: T1-M3-POOL-LISTWISE-RANKING-SFT (M3 SFT improvement, NO RL) ----------
# R15 verdict E (2026-08-29): shortlist oracle recall 0.884 < 0.95; TRUE bottleneck =
# the FROZEN R6 base on the full gated pool (pos-recall@32 = 0.842), not the shortlist
# construction.  R16's ONLY goal: retrain the M3 selector head with a pool-relative
# LISTWISE objective so the frozen M3 itself pushes positive Proposals into the top-32
# of the FULL pool.  NOT a new training phase: same M3Top1Selector proto-head, frozen
# M2/Memory/Reasoner/FixedDecisionReplay, shortlist builder VERBATIM unchanged (only
# the score it ranks by changes via wrapping the new selector in M3RollingGRPOPolicy).
# §6-11: L_total = L_list + lambda_pos·L_posmargin + lambda_pair·L_positive_pair;
# tau_U FIXED (no sweep); U(P) = Cmax(S)-Cmax(S'_P) from frozen local + FDR (TRAIN only).
TO1_R16_ENABLED = True                        # §0 R16 M3-SFT-improvement stage switch
TO1_R16_CKPT = CANONICAL_OUT_DIR / "m3_pool_listwise_sft_v3.pt"           # §37, verdict A ONLY
R16_REPORT = CANONICAL_OUT_DIR / "T1_M3_POOL_LISTWISE_RANKING_SFT_R16_REPORT.md"
TO1_R16_EPOCHS = 15                        # same cadence as TO1_EPOCHS (fixed)
TO1_R16_LR = 1e-3                          # AdamW LR (same as TO1_LR)
TO1_R16_WEIGHT_DECAY = 1e-4
TO1_R16_TAU_U = 2.0                        # §6 q = softmax(U_tilde/tau_U); FIXED (no sweep)
TO1_R16_TAU_P = 1.0                        # §6 p_theta = softmax(logits/tau_p); FIXED
TO1_R16_LAMBDA_POS = 1.0                   # §11 fixed
TO1_R16_LAMBDA_PAIR = 0.5                  # §11 fixed
TO1_R16_POS_MARGIN = 1.0                   # §8 softplus(M - (s_+ - s_0))
TO1_R16_INTERNAL_HELD_COUNT = 2            # §15: TRAIN14 internal-held instances (fixed)
TO1_R16_SELECT_RECALL_K = 20               # §15 epoch selection: held mean pos-recall@K
TO1_R16_RECALL32_GATE = 0.95               # §22 primary target (report only; verdict §33)
TO1_R16_BASE_RECALL32 = 0.8421244056489958 # §22/§33 R15 frozen-base pos-recall@32 anchor
TO1_R16_SIG_RECALL32 = 0.05                # §33 "significantly >0.842" absolute gain floor
TO1_R16_RECALL32_A_FLOOR = 0.90            # verdict A hard floor on pos-recall@32 (report)
TO1_R16_UTIL_MASS_DELTA = 0.05             # §33 "clearly improved" utility-mass@32 floor
TO1_R16_SL_RECALL_GATE = 0.95              # §20 no-oracle shortlist gate (same as §13 tol)

# -- R17 POOL-CONTEXT M3 SFT improvement (T1-M3-POOL-CONTEXT-REPRESENTATION-SFT) -
# R16 (verdict B) proved the listwise LOSS form is effective but head-only retrain on
# FROZEN features caps at rec32 +0.013 / util-mass@32 0.882->0.907.  R17 upgrades the
# M3 PROPOSAL REPRESENTATION (not the loss): PoolContextProposalScorer, still M3-SFT
# improvement, NO RL (§0; §40).  Recorded R16/R15 reference rows (§4, §41) are fixed.
TO1_R17_CKPT = CANONICAL_OUT_DIR / "m3_pool_context_sft_v4.pt"       # §44, verdict A ONLY
R17_REPORT = CANONICAL_OUT_DIR / "T1_M3_POOL_CONTEXT_REPRESENTATION_SFT_R17_REPORT.md"
TO1_R17_EPOCHS = 15                        # same cadence as R16 (§21)
TO1_R17_LR = 1e-3                          # AdamW LR
TO1_R17_WEIGHT_DECAY = 1e-4
TO1_R17_TAU_U = 2.0                        # §16 q = softmax(U_tilde/tau_U); FIXED
TO1_R17_TAU_P = 1.0                        # §16 p = softmax(score/tau_p); FIXED
TO1_R17_LAMBDA_POS = 1.0                   # §21 fixed
TO1_R17_LAMBDA_PAIR = 0.5                  # §21 fixed
TO1_R17_LAMBDA_UTILITY = 0.25              # §20 fixed (no sweep)
TO1_R17_POS_MARGIN = 1.0                   # §17 softplus(M - (s_+ - s_0))
TO1_R17_PROP_HIDDEN = 128                  # §6 encoder width (small capacity)
TO1_R17_ALPHA_CTX = 1.0                    # §26 score = base + alpha_ctx*tanh(delta)
TO1_R17_INTERNAL_HELD_COUNT = 2            # §27: TRAIN14 internal-held instances
TO1_R17_SELECT_K = 32                      # §28 selection: held utility-mass@32
TO1_R17_SL_RECALL_GATE = 0.95              # §33 shortlist best/oracle recall gate
TO1_R17_UTIL_MASS_GATE = 0.95              # §33 shortlist util-mass target (or > R16+delta)
TO1_R17_SELF_FLOOR = 0.95 * 355.0          # §29 no-RL TRAIN closed-loop floor (>=337.25)
TO1_R17_R6_FULL_UM32 = 0.8822254109745098  # §4 R6 full-pool util-mass@32 (R16 report)
TO1_R17_R16_FULL_UM32 = 0.9072527826373459 # §4 R16 full-pool util-mass@32 (R16 report)
TO1_R17_UTIL_ADVANCE = 0.03                # §42 "utility ranking显著改善" over R16 full
TO1_R17_T0_TRAIN = 355.0                   # §4/§29 S0 (R6 M3 SFT full) closed loop
TO1_R17_T1_TRAIN = 272.0                   # §4 R16 listwise head-only full closed loop
TO1_R17_T1_AUX_REAL = 53.0                 # §30 R16 AUX-real held
TO1_R17_T1_AUX_SYN = 70.0                  # §30 R16 AUX-syn held
TO1_R17_T1_VAL = 19.0                      # §39 R16 VAL (final no_grad)
TO1_R17_REC32_BASE = 0.8421244056489958    # §4 R15/R16 frozen-base pos-recall@32 anchor

# -- R18: T1-PROPOSAL-VALIDATION-GATED-JOINT-GRPO ---------------------------------
# R15/R16/R17 暂停。本轮回到 Agent 结构问题：M2 找 root 后先真实 probe（保留 R14 contract，
# G_root>0 -> Tier-A root 永远保留；G_root<=0 才查 Memory -> Tier-B；否则 prune）。Reasoner
# 把 retained roots 展开成 complete legal Proposals 后，不再直接把全部合法 Proposal 交给 M3
# （§13：删除 R15/R16 shortlist32 作为 canonical action space，不再 CAP32）。对每个 complete
# Proposal P 先做 REAL complete-Proposal validation：ProposalProbeGain G_prop(P) =
# Cmax(S_t)-Cmax(S'_P)（FixedDecisionReplay，§4-6）。G_prop>0 -> Proposal Tier-A
# PROVEN_IMMEDIATE_GAIN（uncapped，Memory 无权否决，§6/§8）；G_prop<=0 才查 Memory ->
# Tier-B MEMORY_RESCUED（cap=4，按 state similarity/support/confidence 排序，§10）；否则
# Tier-C PRUNE。M3 action space = ALL validated Tier-A UNION max4 Tier-B UNION STOP（§12）。
# M3 = frozen m3_proposal_top1_sft_v2.pt + small zero-init ProposalEvidenceResidualAdapter
# （输入 gain_norm + is_memory_rescued + memory_confidence，§16-20）；δ=0 时与 R6 SFT
# bit-identical（§20）。M2/M3 Joint GRPO 从 cycle 0 同时开放（§25）；ProposalProbeGain 只 gate
# + observation，绝不进 reward（§22-24）。parent 只用 canonical SFT（§26）。identified=false,
# formal_test_access=0, Formal TEST SEALED。
TO1_R18_PROP_MEMORY_BUDGET = 4            # §10 B_PROP_MEMORY -- Tier-B cap（Tier-A uncapped）
TO1_R18_ALPHA_EVIDENCE = 1.0              # §19 score = base + alpha_evidence*tanh(delta)
TO1_R18_EVID_DIM = 3                      # gain_norm, is_memory_rescued, memory_confidence
TO1_R18_MEM_CONF_SCALE = 2.0              # confidence = ms/(ms+k) ∈ [0,1)（evidence only）
TO1_R18_MEM_SUPPORT_MIN = 2               # Tier-B: min fine/exact support（同 R13 族）
TO1_R18_MEM_SUCCESS_MIN = 0.5             # Tier-B: min historical success rate
TO1_R18_TRAINING_CYCLES = 10              # §48 -- 10 cycles, K=8, H=5, E=3
TO1_R18_GRAPHS_PER_BATCH = 8              # §48 -- 8 graphs/batch（资源不够可 4）
TO1_R18_DELAYED_BENEFIT_SAMPLE = 3        # §44 diagnostic: pruned proposals forwarded per state
# §57 comparison table rows（same RAW ruler; P0/P2 = R14 records cited verbatim）
TO1_R18_P0_FULL_SFT = 355.0               # P0: M2 SFT + M3 SFT + FULL pool NO RL (R14 P0)
TO1_R18_P2_FULL_JOINT = 356.0             # P2: R14 stagewise JOINT full pool (R14 P3)
# verdict thresholds
TO1_R18_COMPRESS_RATIO_GATE = 0.45        # A: mean validated/full < this → significant compression
TO1_R18_DENSE_POOL_MEAN = 32.0            # C: mean validated pool >32 → dense-positive remains
TO1_R18_DENSE_POOL_HIGH = 48.0            # C: p90 validated pool >48 → dense-positive remains
TO1_R18_DELAYED_BENEFIT_MAX = 0.25        # D: delayed_benefit_rate > this → myopic gate harms
TO1_R18_FALSE_RESCUE_MAX = 0.40           # E: memory-rescued selected-terminal-positive fraction
TO1_R18_CKPT = CANONICAL_OUT_DIR / "m2_m3_proposal_validated_joint_grpo_r18.pt"  # §65, verdict A ONLY
R18_REPORT = CANONICAL_OUT_DIR / "T1_PROPOSAL_VALIDATION_GATED_JOINT_GRPO_R18_REPORT.md"

# ---------------------------------------------------------------------------
# R19 -- T1-MULTISTEP-PROPOSAL-VALIDATION-JOINT-GRPO（§10/§20/§28/§36/§56/§57/§66）
# 目标：把 R18 单步硬门升级为 immediate → bounded H=2 rescue → Memory fallback
# ---------------------------------------------------------------------------
TO1_R19_H_VAL = 5                     # §10 有界视野（用户 2026-08-30 覆盖 §10/§11/§66：不强制 H=2；
                                      #   用此前效果最好的 horizon）：proposal P 固定为 step0，
                                      #   续算 ≤ TO1_R13_HORIZON(=5) 步 == 全部先前 JOINT 轮
                                      #   （K=8/H=5/E=3，J0=356、R18 P3=387）的 RL horizon
TO1_R19_PROP_MEMORY_BUDGET = 4        # §20 B_PROP_MEMORY：MEMORY_RESCUED 最多 4
TO1_R19_ALPHA_TEMPORAL = 1.0          # §28 score = base + alpha_temporal*tanh(delta_temporal)
TO1_R19_EVID_DIM = 6                  # §25 g1_norm, gh_norm, is_immediate, is_delayed, is_mem, conf
TO1_R19_MEM_CONF_SCALE = 2.0          # confidence = ms/(ms+k) ∈ [0,1)（evidence only）
TO1_R19_MEM_SUPPORT_MIN = 2           # Memory rescue: min fine/exact support
TO1_R19_MEM_SUCCESS_MIN = 0.5         # Memory rescue: min historical success rate
TO1_R19_TRAINING_CYCLES = 10          # §56 第一版 10 cycles（闸全过直接完整跑，不 sweep）
TO1_R19_GRAPHS_PER_BATCH = 8          # §57 8 graphs/batch（K=8；资源不足 4）
TO1_R19_VALIDATION_POLICY_ID = "r19-frozen:m2-canonical+m3-top1-v2:h5"  # §38 continuation policy id
# §57 parity rows（cited verbatim：同一 raw ruler）
TO1_R19_N1_R18_ONESTEP = 250.0        # N1: R18 one-step validated gate NO RL（R18 实测）
TO1_R19_N0_FULL_SFT = 355.0           # N0: FULL pool M2+M3 SFT NO RL（R14 P0）
TO1_R19_J0_FULL_JOINT = 356.0         # J0: R14 stagewise JOINT full pool（R14 P3）
TO1_R19_J1_ONESTEP_JOINT = 387.0      # J1: R18 one-step validated JOINT（R18 实测）
TO1_R19_R18_DELAYED_0P596 = 0.596     # §47：R18 one-step delayed_benefit ruler（520 traj）
# verdict thresholds
TO1_R19_COMPRESS_RATIO_GATE = 0.45    # A：mean validated/full < → 压缩成立
TO1_R19_DENSE_POOL_MEAN = 32.0        # F：mean validated >32 → 稠密池仍存
TO1_R19_DENSE_POOL_HIGH = 48.0        # F：p90 validated >48 → 稠密池仍存
TO1_R19_DENSE_POOL_HIGH2 = 64.0       # §77：>64 也报告
TO1_R19_RESIDUAL_DELAYED_MAX = 0.25   # A：残差 delayed（>H 才显现）≤0.25 → 视野修复成立
TO1_R19_TEMPORAL_MISS_MAX = 0.25      # A：TEMPORAL_VALIDATION_MISS 显著低于 R18 one-step miss
TO1_R19_ROOT_DELAYED_MAX = 0.25       # E：root_delayed_benefit_rate >0.25 → root myopia 上升为主
TO1_R19_CKPT = CANONICAL_OUT_DIR / "m2_m3_multistep_validated_joint_grpo_r19.pt"  # §79, verdict A ONLY
R19_REPORT = CANONICAL_OUT_DIR / "T1_MULTISTEP_PROPOSAL_VALIDATION_JOINT_GRPO_R19_REPORT.md"

# ---------------------------------------------------------------------------
# R20 -- T1-LEXICOGRAPHIC-PROPOSAL-FALLBACK-JOINT-GRPO（R19 verdict B 的直接后续）
# 目标：把 R19 的 PA∪PB∪PC union 改为 state-level 字典序 fallback（PA>>PB>>PC>>STOP）。
# 禁止：继续扩 validation horizon（R19 已证 tmiss=0）、改 M3 representation、把 GH 加进 reward。
# 唯一改变：Proposal evidence 从 union 变 state-level lexicographic fallback。
# ---------------------------------------------------------------------------
TO1_R20_H_VAL = 2                     # §6 沿用 R19 实测饱和 horizon（H=2 rescue plateau，H=3/5 无增量）
TO1_R20_PROP_MEMORY_BUDGET = 4        # §11 MEMORY_RESCUED cap（沿用 R19）
TO1_R20_ALPHA_TEMPORAL = 1.0          # §17 score = base + alpha*tanh(delta)
TO1_R20_EVID_DIM = 6                  # §16 evid[6]（g1_norm, gh_norm, is_immediate, is_delayed, is_mem, conf）
TO1_R20_MEM_CONF_SCALE = 2.0          # confidence = ms/(ms+k)
TO1_R20_MEM_SUPPORT_MIN = 2
TO1_R20_MEM_SUCCESS_MIN = 0.5
TO1_R20_TRAINING_CYCLES = 10          # §23 沿用 canonical
TO1_R20_GRAPHS_PER_BATCH = 8          # §23 沿用 canonical（K=8；资源不足 4）
TO1_R20_VALIDATION_POLICY_ID = "r20-frozen:m2-canonical+m3-top1-v2:h2"  # §7 冻结 continuation policy id
# §28/§30 cited parity rows（同一 raw ruler，R14/R18/R19 实测，verbatim）
TO1_R20_L0_FULL_SFT = 355.0           # L0: full pool SFT no-RL（R14 P0）
TO1_R20_L1_IMMEDIATE = 250.0          # L1: R18 immediate-only no-RL
TO1_R20_L2_PA_PB = 250.0              # L2: R19 PA+PB union no-RL
TO1_R20_K0_FULL_JOINT = 356.0         # K0: R14 stagewise full Joint
TO1_R20_K1_IMMEDIATE_JOINT = 387.0    # K1: R18 immediate-only Joint
TO1_R20_K2_PA_PB_JOINT = 378.0        # K2: R19 PA+PB Joint
TO1_R20_CKPT = CANONICAL_OUT_DIR / "m2_m3_lexicographic_joint_grpo_r20.pt"  # §53, verdict A ONLY
R20_REPORT = CANONICAL_OUT_DIR / "T1_LEXICOGRAPHIC_PROPOSAL_FALLBACK_JOINT_GRPO_R20_REPORT.md"

# ---------------------------------------------------------------------------
# R21 -- T1-INSTANCE-DIVERSE-JOINT-GRPO（R20 verdict B 的直接后续）
# 目标：R20 的 JOINT 政策只在少数 benchmark TRAIN 图上学到了 instance-specific 残差
# （K3=387，但 AUX-real 7 / AUX-syn 4 / VAL 0）。R21 唯一改变 = Joint GRPO training
# instance distribution（不动环境/动作语义/奖励/架构）。§6 ONLY CHANGE。
# D0 = R20 runtime + 原 benchmark-dominant distribution；D1 = R20 runtime +
# instance-diverse balanced distribution。§13 同 trajectory/environment 交互预算。
# 模型选择 generalization-first（internal-held + AUX-real held + AUX-syn held，
# 源级等权；VAL final-only）。checkpoint 仅 verdict A。
# ---------------------------------------------------------------------------
TO1_R21_TRAINING_CYCLES = 10          # §15 继承 canonical（D0/D1 同 10 轮）
TO1_R21_GRAPHS_PER_BATCH = 8          # §15 继承（gpb 为遗留参数，实际由 per_src/cycle 控制）
TO1_R21_D0_PER_SRC = {"bench": 2, "real": 1, "syn": 1}  # §14 D0 benchmark-dominant（R20 verbatim）
TO1_R21_D1_PER_CYCLE = 4              # §14 D1 unified round-robin 每轮抽 4 个 distinct 实例
TO1_R21_HELD_MARGIN = 2.0             # held-agg 提升显著性绝对下界（配合相对 marg 使用）
TO1_R21_CKPT = CANONICAL_OUT_DIR / "m2_m3_instance_diverse_joint_grpo_r21.pt"  # §49, verdict A ONLY
R21_REPORT = CANONICAL_OUT_DIR / "T1_INSTANCE_DIVERSE_JOINT_GRPO_R21_REPORT.md"

# ---------------------------------------------------------------------------
# T2-A -- MULTI-PATH HELD TRAJECTORY EVALUATION（R21 verdict C 的直接后续）
# EVALUATION-ONLY diagnostic：frozen canonical policy 在 held/unseen 上是否存在
# latent positive trajectory（single greedy 没暴露）。NO training / NO optimizer /
# NO oracle / NO checkpoint promotion。搜索宽度与 worker 并发分开报告。
# canonical K/H/T/eps 沿用 R11-R21 已验证值（§7/§8/§28），不重造、不 sweep。
# ---------------------------------------------------------------------------
TO1_T2A_K = TO1_R13_K                  # §7 canonical sibling width（=8，R11-R21 一贯）
TO1_T2A_HORIZON = TO1_R13_HORIZON      # §8 canonical closed-loop horizon（=5）
TO1_T2A_TEMP = TO1_R13_TEMP            # §11 behavior temperature（=1.5，复用 mixture_sample）
TO1_T2A_MIX_EPS = TO1_R13_MIX_EPS      # §11 uniform mixture mass（=0.10）
TO1_T2A_N_SAMPLED = 8                  # §6 per-root sampled trajectory count（= K）
TO1_T2A_ACTION_SPACE = "lexicographic" # §4 canonical runtime action semantics VERBATIM
TO1_T2A_SEED_BASE = 0                  # §29 deterministic-replay branch seed base
TO1_T2A_FROZEN_ID = "t2a-frozen:m2-canonical+m3-top1-v2:h5:lexicographic"  # §3 frozen policy identity
T2A_REPORT = CANONICAL_OUT_DIR / "T2A_MULTI_PATH_HELD_TRAJECTORY_EVALUATION_REPORT.md"

# ---------------------------------------------------------------------------
# T2-B -- MULTI-PATH JOINT AGENTIC GRPO（T2-A verdict A 的 Stage-3 训练后续）
# 核心问句：K 条同根 sibling 轨迹能否把 T2-A 已证实的 latent positive trajectory
# 训练成 greedy argmax 可提取的高概率行为。reward/advantage/update 与 R21 完全一致
# （lexicographic runtime + stagewise A2/A3）；新增的是训练前后 held 上的
# ExtractionGap / latent / L2G 诊断层，checkpoint 只在 verdict A 写。
# best-of-N 是 POST-HOC 诊断（deployment diagnostic + extraction-gap 测量），
# 绝不进 reward/advantage/oracle。ExtractionGap 用差值（greedy≈0 时比值发散）。
# ---------------------------------------------------------------------------
TO1_T2B_K = TO1_R13_K                  # §5 K=8 siblings per root（R11-R21 一贯）
TO1_T2B_HORIZON = TO1_R13_HORIZON      # §5 H=5
TO1_T2B_TEMP = TO1_R13_TEMP            # §8 exploration T=1.5（复用 mixture_sample）
TO1_T2B_MIX_EPS = TO1_R13_MIX_EPS      # §8 eps=0.10
TO1_T2B_ACTION_SPACE = "lexicographic" # §2 canonical runtime VERBATIM（非 cap32/shortlist32）
TO1_T2B_TRAINING_CYCLES = 10           # R21 scale（single joint stage）
TO1_T2B_GRAPHS_PER_BATCH = 8
TO1_T2B_UPDATE_EPOCHS = TO1_R13_UPDATE_EPOCHS  # E=3 batch reuse（grpo_update_joint）
TO1_T2B_SEED_BASE = 0                  # deterministic replay seed base（复用 t2a_branch_seed）
TO1_T2B_N_SAMPLED = 8                  # best-of-N diagnostic width（== K）
TO1_T2B_DEVICE = "auto"                # "auto"|"cpu"|"cuda"；auto -> cuda iff available（Mac 无 CUDA 回退 cpu）
TO1_T2B_PARENT_ID = "t2b-parent:m2-canonical+m3-top1-v2:h5:lexicographic"  # §3 parent identity（trainable）
TO1_T2B_CKPT = CANONICAL_OUT_DIR / "m2_m3_multipath_joint_grpo_t2b.pt"  # verdict A ONLY
T2B_REPORT = CANONICAL_OUT_DIR / "T2B_MULTI_PATH_JOINT_AGENTIC_GRPO_REPORT.md"
TO1_T2B_GREEDY_HELD_MARGIN = 2.0       # 双闸 gate-1 margin（对齐 TO1_R21_HELD_MARGIN）
TO1_T2B_EXTRACTION_GAP_REDUCE = 0.90   # 双闸 gate-2：final gap <= 0.90 * e0 gap（>=10% 收窄）

# T2-D formal training budget. These expand only the data/interaction budget;
# canonical T2-B reward, action semantics, actor architecture and H=5 stay fixed.
T2D_TRAINING_CYCLES = 100
T2D_GRAPHS_PER_CYCLE = 32
T2D_BRANCHES_PER_GRAPH = 32

# T2-M uses the best makespan found in the remaining horizon as the learning
# target. Regression after that best state remains an observation-only metric;
# its default reward weight is exactly zero.
T2L_NET_REGRESSION_WEIGHT = float(os.environ.get(
    "NET_REGRESSION_WEIGHT", "0.0"))

# T2-L joint-action coverage.  The original implementation advertised mixed
# ROUTE/SEQ families but only materialised ROUTE+ROUTE pairs.  Keep construction
# bounded: rank atomic actions first, then run structural legality only inside
# this compact set.  These are candidate-coverage limits, not reward rules.
T2L_PAIR_ROUTE_ATOMS = int(os.environ.get("PAIR_ROUTE_ATOMS", "24"))
T2L_PAIR_SEQUENCE_ATOMS = int(os.environ.get("PAIR_SEQUENCE_ATOMS", "24"))
T2L_PAIRS_PER_FAMILY = int(os.environ.get("PAIRS_PER_FAMILY", "32"))
T2L_PAIR_TOTAL_CAP = int(os.environ.get("PAIR_TOTAL_CAP", "96"))
T2L_SINGLE_ACTIONS = int(os.environ.get("SINGLE_ACTIONS", "96"))
T2L_BASE_PAIR_ACTIONS = int(os.environ.get("BASE_PAIR_ACTIONS", "64"))
T2L_PLATEAU_PAIR_ACTIONS = int(os.environ.get("PLATEAU_PAIR_ACTIONS", "96"))
T2L_PLATEAU_STREAK = int(os.environ.get("PLATEAU_STREAK", "2"))

# T2-M v3: M2 expands several causal roots per state instead of hard-gating the
# entire downstream search through one root.  M3 shortlist membership is driven
# by the current RL actor; a policy-independent random tail preserves discovery.
T2L_M2_ROOT_TOP_K = int(os.environ.get("M2_ROOT_TOP_K", "6"))
T2L_M2_SIBLINGS_PER_ROOT = int(os.environ.get("M2_SIBLINGS_PER_ROOT", "1"))
T2L_M3_FIRST_STEP_STRATIFIED = bool(int(os.environ.get("M3_FIRST_STEP_STRATIFIED", "1")))
T2L_SHARE_SIBLING_ANALYZE_CACHE = bool(int(os.environ.get("SHARE_SIBLING_ANALYZE_CACHE", "1")))
T2L_ROLLOUT_REPLAY_CACHE_ENTRIES = int(os.environ.get("ROLLOUT_REPLAY_CACHE_ENTRIES", "128"))
T2L_ROLLOUT_REPLAY_CACHE_MB = float(os.environ.get("ROLLOUT_REPLAY_CACHE_MB", "16"))
T2L_M3_POOL_TOP_FRACTION = float(os.environ.get("M3_POOL_TOP_FRACTION", "0.80"))
T2L_ANALYZE_CACHE_ENTRIES = int(os.environ.get("ANALYZE_CACHE_ENTRIES", "16"))
T2L_WORKER_REPLAY_CACHE_ENTRIES = int(os.environ.get("WORKER_REPLAY_CACHE_ENTRIES", "1000"))
T2L_MAX_INFLIGHT_MULTIPLIER = int(os.environ.get("MAX_INFLIGHT_MULTIPLIER", "2"))
T2L_TRAIN_RECORD_FP16 = bool(int(os.environ.get("TRAIN_RECORD_FP16", "1")))
T2L_ANCHOR_TRAJECTORIES = int(os.environ.get("ANCHOR_TRAJECTORIES", "2"))
T2L_ANCHOR_SCAN = int(os.environ.get("ANCHOR_SCAN", "4"))
T2L_ANCHOR_KEEP = int(os.environ.get("ANCHOR_KEEP", "2"))

# -- canonical GRPO (Phase 2) ----------------------------------------------------
GRPO_EPS = 0.2               # PPO clip interval (no sweep)
GRPO_BETA_KL = 0.03          # frozen π_ref KL coefficient
GRPO_ALPHA_RESIDUAL = 0.3    # bounded residual: score_GRPO = score_SFT + alpha*tanh(delta)
GRPO_G_MIN = 8               # group size floor (pad with STOP if fewer actions)
GRPO_ITERS = 24              # real iteration loop (not a sweep)
GRPO_LR = 3e-4
GRPO_GROUP_MAX_MEAN = 16.0   # diagnostic cap: mean group size expected with G=min(8,N)+STOP
