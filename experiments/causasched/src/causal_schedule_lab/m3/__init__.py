"""Patch §7/§11 -- M3 proposal selector over :math:`(S,P,M)`.

Three heads -- ranking / acceptance / risk -- consume a deterministic
``(S,P,M)`` feature vector through a learned memory-feature encoder.  The
Stage-3 loss is :math:`L_{M3}=L_{rank}+L_{accept}+L_{risk}`.

Canonical training modules (T1-M3-CANONICAL-CONSOLIDATION-UTILITY-RERANK-GRPO-R5):
  - :mod:`causal_schedule_lab.m3.proposal_features`  state/proposal features + Frozen-Local exec
  - :mod:`causal_schedule_lab.m3.memory`             replay store + progressive causal memory
  - :mod:`causal_schedule_lab.m3.scorer`             SFT usefulness/rank scorer + pairwise training
  - :mod:`causal_schedule_lab.m3.gate`               ACT/STOP gate (act_primary)
  - :mod:`causal_schedule_lab.m3.ranking`            wide recall + Stage-2 reranker
  - :mod:`causal_schedule_lab.m3.rollout`            replay build + closed-loop rollouts + regressions
  - :mod:`causal_schedule_lab.m3.policy`             canonical GRPO action head + group loss
  - :mod:`causal_schedule_lab.m3.grpo`               canonical vs legacy-GRPO-lite surface
  - :mod:`causal_schedule_lab.m3.upstream`           importlib bridge to the 4 shared scripts
"""

from .grpo import (
    CANONICAL_GRPO,
    LEGACY_GRPO_LITE,
    M3GRPOActionHead,
    grpo_group_loss,
    grpo_group_logp,
    grpo_sample_group,
    m3_canonical_reward,
    m3_grpo_loss,
    m3_grpo_reward_for_group,
    m3_reward,
    resolved_scores,
)
from .selection import SelectionDecision, future_improvement_value, risk_score, select_final
from .selector import (
    DEFAULT_FEATURE_DIM,
    BASE_PROPOSAL_PRUNGS,
    EFFECT_PRUNGS,
    MEMORY_PRUNGS,
    PROPOSAL_PRUNGS,
    STATE_PRUNGS,
    GlobalStateEncoder,
    M3Output,
    M3ProposalSelector,
    M3Targets,
    MemoryEncoder,
    ProposalEncoder,
    m3_input_features,
    m3_base_proposal_features,
    m3_loss,
    m3_memory_features,
    m3_proposal_features,
    m3_state_features,
    memory_prior,
)

__all__ = [
    "DEFAULT_FEATURE_DIM",
    "BASE_PROPOSAL_PRUNGS",
    "EFFECT_PRUNGS",
    "STATE_PRUNGS",
    "PROPOSAL_PRUNGS",
    "MEMORY_PRUNGS",
    "M3Output",
    "M3ProposalSelector",
    "M3Targets",
    "SelectionDecision",
    "GlobalStateEncoder",
    "ProposalEncoder",
    "MemoryEncoder",
    "future_improvement_value",
    "m3_input_features",
    "m3_base_proposal_features",
    "m3_state_features",
    "m3_proposal_features",
    "m3_memory_features",
    "m3_loss",
    "m3_grpo_loss",
    "m3_grpo_reward_for_group",
    "m3_reward",
    "memory_prior",
    "select_final",
    "risk_score",
    "CANONICAL_GRPO",
    "LEGACY_GRPO_LITE",
    "M3GRPOActionHead",
    "grpo_group_loss",
    "grpo_group_logp",
    "grpo_sample_group",
    "m3_canonical_reward",
    "resolved_scores",
]