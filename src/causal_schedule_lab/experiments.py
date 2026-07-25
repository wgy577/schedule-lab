from __future__ import annotations

import math
import random
from statistics import mean, median
from typing import Iterable, Sequence

from .models import CausalInterventionPoint, ExperimentRecord


def core_point_metrics(
    ranked_ids: Sequence[str],
    effective_ids: set[str],
    *,
    k: int,
) -> dict[str, float]:
    top = list(ranked_ids[:k])
    hits = [item in effective_ids for item in top]
    precision = sum(hits) / max(1, len(top))
    recall = sum(hits) / max(1, len(effective_ids))
    dcg = sum(
        int(hit) / math.log2(index + 2)
        for index, hit in enumerate(hits)
    )
    ideal_hits = [True] * min(k, len(effective_ids))
    idcg = sum(
        1.0 / math.log2(index + 2)
        for index, _ in enumerate(ideal_hits)
    )
    return {
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"ndcg@{k}": 0.0 if idcg == 0 else dcg / idcg,
    }


def improvement_metrics(records: Iterable[ExperimentRecord]) -> dict[str, float]:
    records = list(records)
    accepted = [record for record in records if record.accepted]
    deltas = [record.delta_objective for record in accepted]
    runtimes = [record.runtime_seconds for record in records]
    full_oracles = sum(
        any(item.fidelity.value == "full" for item in record.verifications)
        for record in records
    )
    return {
        "candidate_count": float(len(records)),
        "accepted_count": float(len(accepted)),
        "acceptance_rate": len(accepted) / max(1, len(records)),
        "total_improvement": sum(deltas),
        "median_accepted_improvement": median(deltas) if deltas else 0.0,
        "total_runtime": sum(runtimes),
        "improvement_per_second": sum(deltas) / max(sum(runtimes), 1e-9),
        "full_oracle_calls": float(full_oracles),
        "improvement_per_full_oracle": sum(deltas) / max(1, full_oracles),
    }


def build_control_plan(
    candidates: Sequence[CausalInterventionPoint],
    *,
    seed: int = 0,
) -> dict[str, list[dict[str, object]]]:
    """Create the three causal controls; execution remains subject to Oracle budget."""

    rng = random.Random(seed)
    ids = [candidate.id for candidate in candidates]
    random_positions = ids[:]
    rng.shuffle(random_positions)
    same_size = sorted(
        candidates,
        key=lambda item: (
            len(item.closure.operation_ids),
            item.id,
        ),
    )
    return {
        "causal_ranked": [
            {"cip": candidate.id, "operators": list(candidate.recommended_operators)}
            for candidate in candidates
        ],
        "same_operator_random_position": [
            {
                "cip": cip_id,
                "operator": candidates[0].recommended_operators[0]
                if candidates and candidates[0].recommended_operators
                else None,
            }
            for cip_id in random_positions
        ],
        "same_position_random_operator": [
            {
                "cip": candidate.id,
                "operator": (
                    rng.choice(candidate.recommended_operators)
                    if candidate.recommended_operators
                    else None
                ),
            }
            for candidate in candidates
        ],
        "same_closure_size_random_region": [
            {
                "cip": candidate.id,
                "closureSize": len(candidate.closure.operation_ids),
            }
            for candidate in same_size
        ],
    }
