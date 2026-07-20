from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _move_key(incumbent_hash: str, candidate: dict[str, Any]) -> str:
    payload = {
        "incumbent": incumbent_hash,
        "operator": candidate.get("operator"),
        "dispatchSwaps": candidate.get("dispatchSwaps", []),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _beta_summary(successes: int, trials: int, alpha: float = 1.0, beta: float = 1.0) -> dict[str, float]:
    posterior_alpha = alpha + successes
    posterior_beta = beta + trials - successes
    total = posterior_alpha + posterior_beta
    mean = posterior_alpha / total
    variance = posterior_alpha * posterior_beta / (total * total * (total + 1.0))
    return {
        "alpha": posterior_alpha,
        "beta": posterior_beta,
        "mean": mean,
        "standardDeviation": math.sqrt(variance),
        "upper90Approx": min(1.0, mean + 1.645 * math.sqrt(variance)),
    }


def load_evidence(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load and deduplicate deterministic Oracle observations.

    A propagation-closure rerun supersedes the preliminary open-boundary run
    of the same move. Exact reproducibility reruns count once.
    """

    selected: dict[str, dict[str, Any]] = {}
    for path in sorted({str(Path(item).expanduser().resolve()) for item in paths}):
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if "incumbent" not in payload or "candidates" not in payload:
            continue
        incumbent = payload["incumbent"]
        incumbent_hash = str(incumbent["scheduleHash"])
        incumbent_makespan = float(incumbent["trueMakespan"])
        for candidate in payload["candidates"]:
            if not str(candidate.get("operator", "")).startswith("alns-"):
                continue
            swaps = candidate.get("dispatchSwaps", [])
            if not swaps:
                continue
            key = _move_key(incumbent_hash, candidate)
            true_makespan = float(candidate["trueMakespan"])
            observation = {
                "moveKey": key,
                "source": path,
                "incumbentHash": incumbent_hash,
                "incumbentMakespan": incumbent_makespan,
                "operator": str(candidate["operator"]),
                "stage": int(swaps[0]["operation"]) + 1,
                "swapCount": len(swaps),
                "swapDistance": sum(abs(int(item["rightPosition"]) - int(item["leftPosition"])) for item in swaps),
                "neighborhoodSize": len(candidate.get("neighborhoodJobs", [])),
                "trueMakespan": true_makespan,
                "gain": incumbent_makespan - true_makespan,
                "rawImprovement": true_makespan < incumbent_makespan - 1e-6,
                "propagationClosed": bool(candidate.get("propagationClosed", False)),
                "expansionCount": len(candidate.get("requiredExpansionJobs", [])),
                "accepted": bool(candidate.get("accepted", False)),
            }
            previous = selected.get(key)
            quality = (
                int(observation["propagationClosed"]),
                int(observation["accepted"]),
                -observation["expansionCount"],
            )
            previous_quality = (
                int(previous["propagationClosed"]),
                int(previous["accepted"]),
                -previous["expansionCount"],
            ) if previous else None
            if previous is None or quality > previous_quality:
                selected[key] = observation
    return sorted(selected.values(), key=lambda item: (item["operator"], item["moveKey"]))


def rank_operator_evidence(
    observations: list[dict[str, Any]],
    *,
    expansion_penalty: float = 0.15,
    exploration_weight: float = 1.0,
    candidate_operators: Iterable[str] = ("alns-adjacent-o3",),
    retire_after_zero_improvements: int = 4,
) -> dict[str, Any]:
    if not observations:
        raise ValueError("no ALNS Oracle observations were found")
    positive_gains = [float(item["gain"]) for item in observations if item["gain"] > 1e-6]
    prior_gain = statistics.mean(positive_gains) if positive_gains else 1.0
    prior_expansion = statistics.mean(float(item["expansionCount"]) for item in observations)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        grouped[str(observation["operator"])].append(observation)
    for operator in candidate_operators:
        grouped.setdefault(str(operator), [])

    arms = []
    for operator, records in sorted(grouped.items()):
        trials = len(records)
        raw_successes = sum(int(item["rawImprovement"]) for item in records)
        closed_successes = sum(int(item["propagationClosed"]) for item in records)
        accepted_successes = sum(int(item["accepted"]) for item in records)
        raw_posterior = _beta_summary(raw_successes, trials)
        closed_posterior = _beta_summary(closed_successes, trials)
        accepted_posterior = _beta_summary(accepted_successes, trials)
        gains = [float(item["gain"]) for item in records if item["gain"] > 1e-6]
        gain_mean = (2.0 * prior_gain + sum(gains)) / (2.0 + len(gains))
        expansion_mean = (2.0 * prior_expansion + sum(float(item["expansionCount"]) for item in records)) / (
            2.0 + trials
        )
        exploitation = raw_posterior["mean"] * gain_mean - expansion_penalty * expansion_mean
        exploration_bonus = exploration_weight * raw_posterior["standardDeviation"] * gain_mean
        retired = trials >= retire_after_zero_improvements and raw_successes == 0
        priority_score = -1_000_000_000.0 if retired else exploitation + exploration_bonus
        stage = None
        if records and len({item["stage"] for item in records}) == 1:
            stage = records[0]["stage"]
        elif operator.startswith("alns-adjacent-o"):
            stage = int(operator.rsplit("o", 1)[1])
        arms.append(
            {
                "operator": operator,
                "stage": stage,
                "observations": trials,
                "rawImprovementCount": raw_successes,
                "closedCount": closed_successes,
                "acceptedCount": accepted_successes,
                "rawImprovementPosterior": raw_posterior,
                "cheapClosurePosterior": closed_posterior,
                "acceptancePosterior": accepted_posterior,
                "positiveGainPosteriorMean": gain_mean,
                "expectedExpansionJobs": expansion_mean,
                "exploitationScore": exploitation,
                "explorationBonus": exploration_bonus,
                "retired": retired,
                "retirementReason": (
                    f"zero raw improvements after {trials} observations"
                    if retired
                    else None
                ),
                "priorityScore": priority_score,
            }
        )
    arms.sort(key=lambda item: (-item["priorityScore"], item["operator"]))
    return {
        "model": {
            "kind": "deterministic-beta-binomial-plus-shrunk-gain",
            "betaPrior": {"alpha": 1.0, "beta": 1.0},
            "positiveGainPriorMean": prior_gain,
            "positiveGainPriorStrength": 2.0,
            "expansionPenaltyPerJob": expansion_penalty,
            "explorationWeight": exploration_weight,
            "retireAfterZeroImprovements": retire_after_zero_improvements,
            "purpose": "rank Oracle experiments only; never certify feasibility",
        },
        "deduplicatedObservationCount": len(observations),
        "recommendedOperator": arms[0]["operator"],
        "arms": arms,
        "observations": observations,
    }
