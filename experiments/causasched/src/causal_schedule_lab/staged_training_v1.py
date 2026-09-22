"""Staged M2 / trajectory / Effect Predictor / M3 orchestration.

Wires the four stages end-to-end by reuse, not rewrite:

* **Stage 1** -- train M2 with :math:`L_{M2}=L_{root}+\\lambda_1 L_{edit}+\\lambda_2
  L_{dep}`: this is exactly the existing ``compute_m2_loss`` multi-task loss with
  the Patch §4 dependency term active (see Phase A).
* **Stage 2** -- build memory: run the Phase-C :class:`CounterfactualEvaluator`
  over each proposal to materialise :math:`S'=T(S,P)` and store the
  :math:`(S,P,S')` triple.
* **Stage 3** -- train M3 (Phase-D selector) on the memory-produced triples with
  :math:`L_{M3}=L_{rank}+L_{accept}+L_{risk}`.
* **Stage 4** -- joint fine-tune with :math:`L=L_{M2}+\\alpha L_{M3}+\\beta
  L_{global}`; the global term is a plug-in (default ``\\beta=0``).

This module is the *orchestration contract*.  Per the project doctrine it wires
and smoke-tests the loop; it does **not** run a full-scale training sweep and
makes no correctness/improvement claim (`identified=false`, SAFE_TO_TRAIN NO).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor

from .counterfactual import (
    CounterfactualEvaluator,
    CounterfactualResult,
    ProposalEvaluationComparison,
)
from .intervention.effect_predictor_v1 import (
    InterventionEffectPredictor,
    m3_effect_score,
)
from .m2_v5_schema_v1 import LegalEdit
from .m3 import (
    M3ProposalSelector,
    M3Targets,
    m3_loss,
    m3_memory_features,
    m3_base_proposal_features,
    m3_proposal_features,
    m3_state_features,
    memory_prior,
)
from .memory import (
    DIRECT_SUCCESS,
    ProposalRecord,
    encode_state,
    experience_final_gain,
    experience_future_success,
)
from .training.effect_training import freeze_effect_predictor
from .intervention.effect_predictor_v1 import (
    causal_chain_effect_features,
    memory_effect_features,
    proposal_effect_features,
    state_effect_features,
)

__all__ = [
    "STAGE1_WEIGHTS",
    "stage1_m2_loss",
    "stage2_populate_memory",
    "stage3_build_batches",
    "stage3_train",
    "joint_ft_loss",
    "M3Batch",
    "ClosureRuntimeResult",
    "run_m2_m3_closure",
]

# Stage-1 weights: L_M2 = L_root + λ1·L_edit + λ2·L_dep (Patch §8).
STAGE1_WEIGHTS = {
    "w_root": 1.0,
    "w_edit": 1.0,
    "w_dep": 1.0,
    # Existing relevance heads remain optional diagnostics, not hidden terms in
    # the frozen Stage-1 objective.
    "w_node": 0.0,
    "w_edge": 0.0,
    "w_rel": 0.0,
}


def stage1_m2_loss(out, targets, **override: float) -> dict[str, Tensor]:
    """Stage-1 M2 loss (Patch §8) -- the V5 multi-task loss incl. L_dep."""
    from .m2_training_v1 import compute_m2_loss

    weights = {**STAGE1_WEIGHTS, **override}
    return compute_m2_loss(out, targets, **weights)


def stage2_populate_memory(
    problem,
    incumbent,
    *,
    store,
    proposals: Sequence[tuple[ProposalRecord, Sequence[LegalEdit]]],
    solver_time: float = 1.0,
    theta: float = 0.1,
    store_outcome: bool = True,
) -> dict[str, CounterfactualResult]:
    """Stage 2 -- materialise + store :math:`S'=T(S,P)` for each proposal."""
    evaluator = CounterfactualEvaluator(
        store, solver_time=solver_time, theta=theta
    )
    outcomes: dict[str, CounterfactualResult] = {}
    for proposal, edits in proposals:
        res = evaluator.evaluate(
            problem, incumbent, proposal, edits, store_outcome=store_outcome
        )
        outcomes[proposal.proposal_id or res.key] = res
    return outcomes


@dataclass
class M3Batch:
    """A Stage-3 batch: three (S,P,M) prong matrices + supervision for one state."""

    state_f: Tensor    # [n, STATE_PRUNGS]
    prop_f: Tensor     # [n, PROPOSAL_PRUNGS]
    mem_f: Tensor      # [n, MEMORY_PRUNGS]
    targets: M3Targets    # rank_order indices best-first + accept/risk per row


@dataclass
class ClosureRuntimeResult:
    """Evidence from Schedule→M2→Effect→M3→final CP-SAT validation."""

    m2_output: object
    evaluated_proposals: tuple[object, ...]
    counterfactual_results: tuple[CounterfactualResult, ...]
    counterfactual_comparisons: tuple[ProposalEvaluationComparison, ...]
    m3_scores: tuple[float, ...]
    acceptance: tuple[float, ...]
    risk: tuple[float, ...]
    predicted_success: tuple[float, ...]
    predicted_gain: tuple[float, ...]
    predicted_fiv: tuple[float, ...]
    predicted_delta_cmax: tuple[float, ...]
    predicted_effect_risk: tuple[float, ...]
    candidate_pool_size: int
    chosen_index: int | None
    status: str = "M2/M2.5/M3 Runtime Integrated"


def run_m2_m3_closure(
    problem,
    schedule,
    appearance,
    *,
    selector: M3ProposalSelector,
    effect_predictor: InterventionEffectPredictor,
    store,
    model=None,
    max_proposals: int = 3,
    proposal_pool_limit: int = 100,
    solver_time: float = 0.5,
    theta: float = 0.1,
) -> ClosureRuntimeResult:
    """Predict/rank a proposal pool, then CP-SAT-validate only the final set."""
    from .counterfactual import proposal_from_causal
    from .sg_sct_model_v5 import run_m2_v5_schedule

    out, _model, _bundle = run_m2_v5_schedule(
        problem,
        schedule,
        appearance,
        case_id=problem.id,
        model=model,
    )
    ranked = sorted(out.proposals, key=lambda p: p.proposal_score, reverse=True)
    pool = tuple(ranked[: max(0, proposal_pool_limit)])
    evaluator = CounterfactualEvaluator(
        store, solver_time=solver_time, theta=theta
    )
    state = encode_state(problem, schedule)
    trace_by_proposal = dict(getattr(out, "causal_search_traces", ()))
    def matching_chain(proposal):
        candidates = [chain for chain in out.causal_explanation_chains
                      if chain.appearance_id == proposal.appearance_id
                      and tuple(chain.nodes) == tuple(proposal.causal_chain)]
        return candidates[0] if candidates else None

    chains = [matching_chain(proposal) for proposal in pool]
    records = [proposal_from_causal(
        proposal,
        causal_search_trace=trace_by_proposal.get(proposal.proposal_id, ()),
        causal_chain=chains[index],
    ) for index, proposal in enumerate(pool)]
    priors = [
        memory_prior(store, state, query_proposal=record)
        for record in records
    ]
    effect_predictor.eval()
    effect_predictor.memory_store = store
    selector.eval()
    if pool:
        selector_device = next(selector.parameters()).device
        effect_states = [encode_state(
            problem, schedule, appearance_type=record.appearance_type,
            appearance_score=record.confidence,
        ) for record in records]
        memory_features = [
            m3_memory_features(estimate=p[0], nv=p[1], mean_gain=p[2])
            for p in priors
        ]
        predictions = [effect_predictor.predict(
            effect_states[index], chains[index] or records[index], records[index]
        ) for index in range(len(records))]
        delta_all = [prediction.delta_cmax_pred for prediction in predictions]
        success_all = [prediction.success_probability for prediction in predictions]
        risk_pred_all = [prediction.risk for prediction in predictions]
        gain_all = [prediction.fiv_pred / max(prediction.success_probability, 1e-12)
                    for prediction in predictions]
        fiv_all = [prediction.fiv_pred for prediction in predictions]
        m3_state = torch.tensor(
            [m3_state_features(item) for item in effect_states],
            dtype=torch.float32, device=selector_device,
        )
        m3_proposal = torch.tensor([
            m3_proposal_features(
                record,
                predicted_delta_cmax=delta_all[index],
                predicted_success_probability=success_all[index],
                predicted_fiv=fiv_all[index],
                predicted_risk=risk_pred_all[index],
            )
            for index, record in enumerate(records)
        ], dtype=torch.float32, device=selector_device)
        m3_memory = torch.tensor(
            memory_features, dtype=torch.float32, device=selector_device
        )
        with torch.no_grad():
            m3_out = selector(m3_state, m3_proposal, m3_memory)
        # Until a formal M3 checkpoint exists, shortlist ranking follows the
        # frozen, auditable effect formula rather than random neural logits.
        adjusted_all = [m3_effect_score(prediction) for prediction in predictions]
        accepts_all = [float(value) for value in m3_out.accept_prob]
        risks_all = [float(value) for value in m3_out.risk_prob]
    else:
        delta_all, success_all, risk_pred_all, gain_all, fiv_all = [], [], [], [], []
        adjusted_all, accepts_all, risks_all = [], [], []

    validation_indices = sorted(
        range(len(pool)), key=lambda index: adjusted_all[index], reverse=True
    )[: max(0, max_proposals)]
    selected = tuple(pool[index] for index in validation_indices)
    results: list[CounterfactualResult] = []
    comparisons: list[ProposalEvaluationComparison] = []
    for index in validation_indices:
        proposal, record, prior = pool[index], records[index], priors[index]
        comparison = evaluator.evaluate_dual(
            problem, schedule, record, proposal.edits, store_outcome=True,
            causal_chain=chains[index], root=record.root_decision_id,
            transition_trace=trace_by_proposal.get(proposal.proposal_id, ()),
        )
        comparisons.append(comparison)
        assert comparison.local_counterfactual is not None
        results.append(comparison.local_counterfactual)
    scores = [adjusted_all[index] for index in validation_indices]
    accepts = [accepts_all[index] for index in validation_indices]
    risks = [risks_all[index] for index in validation_indices]
    predicted_success = [success_all[index] for index in validation_indices]
    predicted_gain = [gain_all[index] for index in validation_indices]
    predicted_fiv = [fiv_all[index] for index in validation_indices]
    # Deterministic evaluator acceptance is authoritative.  The learned heads
    # are predictions/diagnostics and cannot veto a direct Cmax improvement.
    admissible = [i for i, result in enumerate(results) if result.keep]
    chosen = min(
        admissible,
        key=lambda i: (
            0 if results[i].classification == DIRECT_SUCCESS else 1,
            -scores[i],
            results[i].risk,
        ),
        default=None,
    )
    return ClosureRuntimeResult(
        m2_output=out,
        evaluated_proposals=selected,
        counterfactual_results=tuple(results),
        counterfactual_comparisons=tuple(comparisons),
        m3_scores=tuple(scores),
        acceptance=tuple(accepts),
        risk=tuple(risks),
        predicted_success=tuple(predicted_success),
        predicted_gain=tuple(predicted_gain),
        predicted_fiv=tuple(predicted_fiv),
        predicted_delta_cmax=tuple(delta_all[index] for index in validation_indices),
        predicted_effect_risk=tuple(risk_pred_all[index] for index in validation_indices),
        candidate_pool_size=len(pool),
        chosen_index=chosen,
    )


def stage3_build_batches(
    store, *, effect_predictor: InterventionEffectPredictor,
    k: int = 5, risk_threshold: float = 0.5,
) -> list[M3Batch]:
    """Group resolved memory by state; build (S,P,M) prongs + targets.

    Labels are proposal-specific ground truth: 2 iff immediate ΔCmax<0; 1 iff
    immediate ΔCmax=0 and its observed trajectory eventually lowers Cmax with
    bounded risk; otherwise 0.  Predicted effect is an input, never the label.
    """
    grouped: dict[str, list] = defaultdict(list)
    from .memory import experience_training_eligible
    for rec in store.resolved():
        if not experience_training_eligible(rec):
            continue
        grouped[json.dumps(asdict(rec.state), sort_keys=True)].append(rec)

    freeze_effect_predictor(effect_predictor)
    effect_device = next(effect_predictor.parameters()).device
    batches: list[M3Batch] = []
    for _sv, recs in grouped.items():
        s_feats: list[tuple[float, ...]] = []
        p_feats: list[tuple[float, ...]] = []
        m_feats: list[tuple[float, ...]] = []
        accepts: list[float] = []
        risks: list[float] = []
        classes: list[int] = []
        rank_keys: list[tuple[float, float, float]] = []
        for r in recs:
            # Leave-one-out: a training row can never retrieve itself.
            prior = memory_prior(
                store,
                r.state,
                k=k,
                query_proposal=r.proposal,
                exclude_key=r.key,
            )
            s_feats.append(m3_state_features(r.state))
            memory_feature = m3_memory_features(
                estimate=prior[0], nv=prior[1], mean_gain=prior[2], k=k
            )
            with torch.no_grad():
                effect = effect_predictor(
                    torch.tensor([state_effect_features(r.state)], dtype=torch.float32, device=effect_device),
                    torch.tensor([causal_chain_effect_features(r.proposal)], dtype=torch.float32, device=effect_device),
                    torch.tensor([proposal_effect_features(r.proposal)], dtype=torch.float32, device=effect_device),
                    torch.tensor([memory_effect_features({
                        "estimate": prior[0], "success_probability": prior[0],
                        "nv": prior[1], "mean_gain": prior[2],
                        "fiv": prior[0] * prior[2], "expected_value": 0.0,
                        "mean_risk": 0.0,
                    }, k=k)], dtype=torch.float32, device=effect_device),
                )
            p_feats.append(m3_proposal_features(
                r.proposal,
                predicted_delta_cmax=float(effect.delta_cmax_pred.item()),
                predicted_success_probability=float(effect.success_prob.item()),
                predicted_fiv=float(effect.fiv.item()),
                predicted_risk=float(effect.risk.item()),
            ))
            m_feats.append(memory_feature)
            outcome = r.outcome
            delta = float(outcome.delta_cmax) if outcome is not None else 0.0
            risk = float(outcome.risk) if outcome is not None else 1.0
            future_success = experience_future_success(r)
            final_gain = experience_final_gain(r)
            if delta < -1e-9:
                label, metric = 2, -delta
            elif abs(delta) <= 1e-9 and future_success and risk <= risk_threshold:
                label, metric = 1, final_gain
            else:
                label, metric = 0, 0.0
            accepts.append(1.0 if label > 0 else 0.0)
            risks.append(max(0.0, min(1.0, risk)))
            classes.append(label)
            rank_keys.append((-float(label), -float(metric), risk))
        if not s_feats:
            continue
        batches.append(M3Batch(
            state_f=torch.tensor(s_feats, dtype=torch.float32),
            prop_f=torch.tensor(p_feats, dtype=torch.float32),
            mem_f=torch.tensor(m_feats, dtype=torch.float32),
            targets=M3Targets(
                accept=(accepts[0] if len(accepts) == 1 else tuple(accepts)),
                risk=(risks[0] if len(risks) == 1 else tuple(risks)),
                outcome_class=tuple(classes),
                rank_order=tuple(sorted(range(len(recs)), key=rank_keys.__getitem__)),
            ),
        ))
    return batches


def stage3_train(
    selector: M3ProposalSelector,
    store,
    *,
    effect_predictor: InterventionEffectPredictor,
    steps: int = 1,
    lr: float = 1e-2,
    k: int = 5,
) -> dict[str, float]:
    """Stage-3 M3 training over the memory-built batches (Patch §8)."""
    batches = stage3_build_batches(store, effect_predictor=effect_predictor, k=k)
    if not batches:
        return {}
    opt = torch.optim.Adam(selector.parameters(), lr=lr)
    history: list[float] = []
    for _ in range(max(1, steps)):
        total = torch.zeros((), dtype=torch.float32)
        for b in batches:
            total = total + m3_loss(
                selector, b.state_f, b.prop_f, b.mem_f, b.targets
            )["total"]
        opt.zero_grad()
        total.backward()
        opt.step()
        history.append(float(total.detach()))
    return {"loss_first": history[0], "loss_last": history[-1], "nb": len(batches)}


def joint_ft_loss(
    m2_terms: Mapping[str, Tensor],
    m3_total: Tensor,
    *,
    effect_total: Tensor | None = None,
    alpha: float = 1.0,
    effect_weight: float = 1.0,
    beta: float = 0.0,
) -> dict[str, Tensor]:
    """Stage-4 joint loss for M2 + Effect Predictor + M3.

    Stage 2 trains Effect separately and Stage 3 freezes it.  This optional
    Stage-4 composition is the only place its supervised trajectory loss may be
    combined with M2/M3; no optimizer is created here.
    """
    m2 = sum(m2_terms.values()) if m2_terms else m3_total.new_zeros(())
    effect = (
        effect_weight * effect_total
        if effect_total is not None else m3_total.new_zeros(())
    )
    parts = {
        "m2": m2,
        "effect": effect,
        "m3": alpha * m3_total,
        "total": m2 + effect + alpha * m3_total,
    }
    if beta:
        # global term (objective alignment) is provided by the caller as a tensor
        # in ``m3_total`` scaling is above; here we reserve beta·L_global as 0
        # unless the caller passes ``beta`` with an explicit global source.
        parts["global"] = beta * m3_total.new_zeros(())
    return parts
