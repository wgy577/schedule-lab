"""Patch §5 -- intervention Experience Memory (S, P, S').

The memory stores **state-transition experiences**, not bare action records::

    S  -- the pre-intervention scheduling state (graph/appearance embeddings when
          available + Cmax + critical machine + load/gap features)
    P  -- the intervention proposal (root cause + intervention graph +
          dependency graph)
    S' -- the post-intervention outcome (delta Cmax, diagnostic delta gap,
          side-effect risk, outcome class), optionally continued as a real
          multi-step trajectory whose endpoint supplies future value

Layers (Patch §11 ``memory/``):

* :mod:`.state_encoder` -- deterministic schedule-state features + a fixed-length
  numeric state vector used for similarity retrieval.
* :mod:`.experience_store` -- atomic in-memory (optionally JSON-persisted)
  store of :class:`InterventionExperience` records, deduped by key.
* :mod:`.retrieval` -- retrieve past transitions whose state is most similar to a
  query (API consumed by M3's memory prior and by the counterfactual evaluator).

Torch-free at the record level (mirrors the V5 schema doctrine); a learned
memory feature encoder for the M3 heads is added in Patch Phase-D.
"""

from .experience_store import (
    DELAYED_SUCCESS,
    DIRECT_SUCCESS,
    FAILURE,
    PENDING,
    ExperienceStore,
    InterventionExperience,
    Outcome,
    ProposalRecord,
    TrajectoryStep,
    TrajectoryRecord,
    experience_final_gain,
    experience_training_eligible,
    experience_future_success,
)
from .retrieval import (
    experience_similarity,
    memory_success_estimate,
    retrieve_experiences,
    top_k_by_similarity,
)
from .similarity import (
    composite_similarity,
    memory_effect_prior,
    memory_prior_composite,
    retrieve_similar,
    retrieve_similar_scored,
    sim_appearance,
    sim_graph,
    sim_proposal,
)
from .state_encoder import StateFeatures, encode_state, state_vector

__all__ = [
    "DIRECT_SUCCESS",
    "DELAYED_SUCCESS",
    "FAILURE",
    "PENDING",
    "StateFeatures",
    "encode_state",
    "state_vector",
    "InterventionExperience",
    "Outcome",
    "ProposalRecord",
    "TrajectoryStep",
    "TrajectoryRecord",
    "experience_future_success",
    "experience_final_gain",
    "experience_training_eligible",
    "ExperienceStore",
    "retrieve_experiences",
    "top_k_by_similarity",
    "experience_similarity",
    "memory_success_estimate",
    "composite_similarity",
    "retrieve_similar",
    "memory_prior_composite",
    "memory_effect_prior",
    "retrieve_similar_scored",
    "sim_graph",
    "sim_appearance",
    "sim_proposal",
]
