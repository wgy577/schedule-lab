"""Patch §6 -- Counterfactual evaluation :math:`S'=T(S,P)`.

The evaluator executes an intervention proposal's editing steps with the
deterministic CP-SAT transition, classifies the outcome (direct success /
delayed success / failure), and records the :math:`(S,P,S')` outcome into the
Phase-B :class:`~causal_schedule_lab.memory.ExperienceStore`.
"""

from .evaluator import (
    DELAYED_SUCCESS,
    DIRECT_SUCCESS,
    FAILURE,
    PENDING,
    CounterfactualEvaluator,
    CounterfactualResult,
    FrozenEvaluationResult,
    GlobalEvaluationResult,
    ProposalEvaluationComparison,
    FROZEN_LOCAL,
    FREE_GLOBAL,
    LOCAL_COUNTERFACTUAL_INFEASIBLE,
    classify_intervention,
    intervention_risk,
    load_imbalance,
    mean_gap,
    plan_from_legal_edits,
    proposal_from_edits,
    proposal_from_causal,
)

__all__ = [
    "CounterfactualEvaluator",
    "CounterfactualResult",
    "FrozenEvaluationResult",
    "GlobalEvaluationResult",
    "ProposalEvaluationComparison",
    "FROZEN_LOCAL",
    "FREE_GLOBAL",
    "LOCAL_COUNTERFACTUAL_INFEASIBLE",
    "classify_intervention",
    "intervention_risk",
    "plan_from_legal_edits",
    "proposal_from_edits",
    "proposal_from_causal",
    "mean_gap",
    "load_imbalance",
    "DELAYED_SUCCESS",
    "DIRECT_SUCCESS",
    "FAILURE",
    "PENDING",
]
